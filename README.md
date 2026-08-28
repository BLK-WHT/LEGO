# LEGO: Synergizing Expert GraphRAG and Expert Chain-of-Thought for Legal Reasoning

## Requirements

```bash
pip install -r requirements.txt
```

Two OpenAI-compatible endpoints are required — a chat model and an embedding
model. Any server works (vLLM, SGLang, a hosted relay).

```bash
export LLM_BASE_URL=...    LLM_API_KEY=...    LLM_MODEL=Qwen3-8B
export EMBED_BASE_URL=...  EMBED_API_KEY=...  EMBED_MODEL=Qwen3-Embedding-8B
```

The reader runs with thinking mode off.

## Data

Everything below ships with the repository.


| Path                    |                                                                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `data/items.jsonl`      | 723 civil law multiple-choice items — `id`, `case`, `question`, `options`, `answer`, `hops`                                               |
| `data/civil_code.jsonl` | 1,260 Civil Code articles — `article_no`, `title`, `text`, and the `book` / `subbook` / `chapter` / `section` headings that aid embedding |
| `data/graph/`           | the expert provision graph — see below                                                                                                    |


`hops` lists the Civil Code articles the official rationale cites for an item, so
its length is that item's hop count — the number of provisions the answer turns
on. 342 items turn on one article, 215 on two, 104 on three, and 62 on four or
more. Nothing at inference time reads it; it is there to stratify results by
reasoning depth and to score retrieval against gold provisions.

To score your own items, supply the same fields; `answer` may be omitted, and the
run is then unscored. `hops` is optional throughout.

## Evaluation

```bash
python run.py --items data/items.jsonl --articles data/civil_code.jsonl --out lego.jsonl
python run.py --items data/items.jsonl --method zeroshot --out zeroshot.jsonl
```

## ExpertGraph

`data/graph/` holds 318 concepts over 409 cards, 4,152 support edges and 1,211
articles. A card states one doctrinal rule — its issue frames, positive and
negative conditions, legal effects and exceptions — and every field is embedded,
so a query routes to concepts rather than to article text. Edges carry the
support strength of a concept for an article, learned by a six-view alignment
(semantic, field-level, structural, hierarchical, and two reference views) and
smoothed by entropic optimal transport. A concept may carry several cards; at
retrieval it competes on its best-matching one.

## Repository layout

```text
lego/
  graph.py       concept cards and their support edges
  retrieval.py   query routing, then coverage-aware greedy selection
  reader.py      the P-F-C prompt, the closed-book prompt, answer parsing
  corpus.py      Civil Code articles
  client.py      OpenAI-compatible chat and embedding gateways
data/
  items.jsonl        723 questions
  civil_code.jsonl   1,260 articles
  graph/             the expert provision graph
run.py           entry point for both methods
```

