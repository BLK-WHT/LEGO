from __future__ import annotations

import re
from dataclasses import dataclass

from .client import chat
from .corpus import Article

SYSTEM_PROMPT = (
    "你是严谨的中国民法客观题助手。"
    "你必须严格基于题干事实、选项和给定法条推理，不得编造未给出的事实或法条。"
)

USER_PROMPT = """请使用 P/F/C（Provision-Fact-Conclusion）方法解答这道中国民法客观题。

角色：站在中立裁判者/考试阅卷者立场。
要求：
1. 先用一句话明确本题要判断的法律对象和题目问法。
2. P-Provision：只列出【给定法条】中真正控制本题的裁判规则；不要新增未给出的法条。
3. P 阶段必须说明法条关系：一般规则/特别规则/但书例外/限制条件/救济条款之间谁控制结论。
4. 特别注意区分：合同效力、物权变动、对抗效力、责任承担、损害赔偿、优先权/抗辩权，不能把一种法律后果扩大成另一种。
5. 遇到“不得仅以……认定无效”“不影响……效力”“可以请求赔偿”等表述，应按其限制功能理解，不得反向推出合同无效或权利当然消灭。
6. F-Fact：列出会影响判断的题干事实，不要写终局法律结论。
7. C-Conclusion：先做一次“法条关系校验”，再逐项判断 A/B/C/D 的选项命题是否成立。每个选项最多一句判断。
8. Conclusion 必须根据题目问法输出应选项；如果题干问“错误的是/不正确的是/不合法的是/不能成立的是”，应选择错误项。
9. 不要为了覆盖而多选；只有选项的主体、法律谓词、要件事实和法律后果都完整匹配时才选。
10. 每一段用 1-3 句，重点比较选项，不要长篇复述。
11. 最后一行必须且只能是 [正确答案]字母<eoa>，例如 [正确答案]A<eoa> 或 [正确答案]BD<eoa>。

【给定法条】
{provisions}

【案情】
{case}

【问题/选项】
{question}

请按以下格式输出：
判断对象:

P-Provision:
法条关系:

F-Fact:


C-Conclusion:
法条关系校验:

[正确答案]...<eoa>"""

ZEROSHOT_SYSTEM = "你是一个严谨的中国法律选择题助手，必须严格遵守作答格式要求。"

ZEROSHOT_PROMPT = """请你运用法律知识从 A、B、C、D 中作答，将你的选项字母写在[正确答案]和<eoa>之间，例如[正确答案]A<eoa>、[正确答案]BD<eoa>、[正确答案]ABCD<eoa>（连续书写、不加空格，顺序为 A-D）。请你严格按照这个格式回答。

### 案情
{case}

### 问题
{question}

### 选项
{options}

### 作答
"""

_TAGGED = re.compile(r"\[\s*正确答案\s*\](.*?)<\s*eoa\s*>", re.IGNORECASE | re.DOTALL)
_PLAIN = re.compile(r"(?:^|\n)\s*(?:最终答案|正确答案|答案)\s*[:：]\s*([A-Da-d\s,、]+)")

_MARKER = re.compile(r"(?:^|[\s　]|(?<=[。；，,!！?？]))([ABCD])\s*[.\.、)）]")
_BOUNDARY = re.compile(r"(?:^|[\s　]|(?<=[。；，,!！?？]))([ABCD])\s*[.\.、)）][\s　]*")
_BOUNDARY_WS = re.compile(r"(?:^|[\s　])([ABCD])\s*[.\.、)）][\s　]*")


def _split(text: str, boundary: re.Pattern, last_wins: bool) -> dict[str, str]:
    matches = list(boundary.finditer(text))
    out: dict[str, str] = {}
    for i, match in enumerate(matches):
        letter = match.group(1)
        if not last_wins and letter in out:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[letter] = text[match.end() : end].strip(" 　\n\t")
    return out


def format_options(options: str) -> str:
    text = (options or "").strip()
    if not text:
        return text
    letters = _MARKER.findall(text)
    if not letters:
        return text
    duplicated = len(letters) != len(set(letters))
    split = _split(text, _BOUNDARY if duplicated else _BOUNDARY_WS, duplicated)
    if len(split) < 2:
        return text
    lines = [f"{letter}. {split[letter]}" for letter in "ABCD" if letter in split]
    return "\n".join(lines) if lines else text


def query_text(case: str, question: str, options: str) -> str:
    return "\n".join(
        [f"案情：{case.strip()}", f"问题：{question.strip()}", f"选项：{format_options(options)}"]
    )


@dataclass
class Prediction:
    answer: set[str]
    response: str


def format_provisions(articles: list[Article]) -> str:
    return "\n".join(
        f"[{rank}] 《民法典》第{a.no}条：{a.as_passage()}".replace("\n", " ")
        for rank, a in enumerate(articles, 1)
    )


def parse_answer(response: str) -> set[str]:
    for pattern in (_TAGGED, _PLAIN):
        matches = pattern.findall(response or "")
        if matches:
            return set(re.findall(r"[A-D]", matches[-1].upper()))
    return set()


def answer_without_provisions(
    case: str, question: str, options: str, max_tokens: int = 512
) -> Prediction:
    prompt = ZEROSHOT_PROMPT.format(
        case=case.strip(), question=question.strip(), options=format_options(options)
    )
    response = chat(ZEROSHOT_SYSTEM, prompt, max_tokens=max_tokens)
    return Prediction(answer=parse_answer(response), response=response)


def answer(
    case: str, question: str, options: str, articles: list[Article], max_tokens: int = 1800
) -> Prediction:
    prompt = USER_PROMPT.format(
        provisions=format_provisions(articles),
        case=case.strip(),
        question=f"{question.strip()}\n\n选项:\n{format_options(options)}",
    )
    response = chat(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)
    return Prediction(answer=parse_answer(response), response=response)
