from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Article:
    no: int
    title: str
    text: str
    book: str = ""
    subbook: str = ""
    chapter: str = ""
    section: str = ""

    def as_passage(self) -> str:
        heading = " / ".join(
            p for p in (self.book, self.subbook, self.chapter, self.section) if p
        )
        head = f"{heading} · {self.title}" if heading else self.title
        return f"【{head}】\n{self.text}"


def load_articles(path: str | Path, max_article_no: int = 1260) -> list[Article]:
    articles: dict[int, Article] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            no = int(row["article_no"])
            if not 0 < no <= max_article_no or no in articles:
                continue
            articles[no] = Article(
                no=no,
                title=row.get("title") or f"第{no}条",
                text=row.get("text", ""),
                book=row.get("book", ""),
                subbook=row.get("subbook", ""),
                chapter=row.get("chapter", ""),
                section=row.get("section", ""),
            )
    return [articles[no] for no in sorted(articles)]
