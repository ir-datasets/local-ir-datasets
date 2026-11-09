# Local ir_datasets

This is an ir_datasets provider that manages access to locally provided datasets.

One application of this provider is to use it with a small set of corpora as part of an IR course.

## Installation

```bash
pip install .          # core (ir_datasets v2 and huggingface_hub/datasets are expected to be installed)
pip install ".[full]"  # additionally installs tira
```

## `hf-local:` provider

`hf-local:` loads a dataset from a local directory through `ir_datasets.v2`, with no network access and no Hugging Face Hub repo. It reuses the real `hf:` card parsing and file reading, so a directory that works here also works as a published `hf:` dataset.

### 1. Lay out the directory

The directory must look like a Hugging Face dataset repo: a `README.md` with an `ir_datasets:` YAML front matter card, plus the data files the card references (paths are relative to the directory):

```
my-dataset/
├── README.md
└── data/
    ├── docs.jsonl      {"doc_id": "d1", "text": "..."}
    ├── queries.jsonl   {"query_id": "q1", "text": "..."}
    └── qrels.jsonl     {"query_id": "q1", "doc_id": "d1", "relevance": 1}
```

```yaml
---
ir_datasets:
  - fragment: null            # the default entry, loaded when no fragment is given
    type: irds:AdhocBenchmark
    docs: docs
    queries: queries
    qrels: qrels
  - fragment: docs
    type: irds:Docs
    file: data/docs.jsonl
  - fragment: queries
    type: irds:Queries
    file: data/queries.jsonl
  - fragment: qrels
    type: irds:Qrels
    file: data/qrels.jsonl
---
```

A complete example is in [`tests/resources/example-hf-dataset`](tests/resources/example-hf-dataset).

### 2. Load it

Importing `local_ir_datasets` registers the `hf-local:` provider:

```python
import ir_datasets.v2
import local_ir_datasets  # registers "hf-local:"

# The whole benchmark (the card's default entry)
benchmark = ir_datasets.v2.load("hf-local:tests/resources/example-hf-dataset")
for doc in benchmark.docs: ...
for query in benchmark.queries: ...
for qrel in benchmark.qrels: ...

# A single table, selected by the card's `fragment`
docs = ir_datasets.v2.load("hf-local:tests/resources/example-hf-dataset/docs")
```

Address format: `hf-local:<path>[/fragment]`.
- `<path>` is relative to the current working directory, or absolute.
- The optional `/fragment` selects one entry of the card. The whole spec is first tried as a directory; only if that is not a directory containing a `README.md` is the last segment treated as the fragment.
- Loading the same path twice returns the same object.
- A path that is neither such a directory nor a fragment of one raises an error.

### Redirect an existing `hf:` id

To make one existing `hf:<repo>` id resolve to a local directory for a limited time (for example to test code that hard-codes the id), use the context manager:

```python
from local_ir_datasets import local_hf_repo

with local_hf_repo("some-org/some-dataset", "path/to/local/dir"):
    benchmark = ir_datasets.v2.load("hf:some-org/some-dataset")
```

Outside the `with` block the id resolves against the real Hub again. Other `hf:` ids are never affected.

Both mechanisms are workarounds until `ir_datasets` supports local `hf:` repos natively.

## TIRA-format directories

A directory with `queries.jsonl`, `corpus.jsonl.gz` (or `documents.jsonl[.gz]`) and optionally `qrels.txt` (see [`tests/resources/example-tira-dataset`](tests/resources/example-tira-dataset)) can be loaded directly with `tira`:

```python
from tira.third_party_integrations import ir_datasets

dataset = ir_datasets.load("tests/resources/example-tira-dataset")
docs, queries, qrels = dataset.docs_iter(), dataset.queries_iter(), dataset.qrels_iter()
```

## Dataset aliases

`src/local_ir_datasets/aliases.py` holds `DATASET_ALIASES`, a dictionary from short dataset ids to locations. It is empty by default. Add entries to use short ids:

```python
DATASET_ALIASES = {"pangrams": "tests/resources/example-hf-dataset"}
```

```python
ir_datasets.v2.load("hf-local:pangrams")        # same as hf-local:<path>
ir_datasets.v2.load("hf-local:pangrams/docs")   # with a fragment
local_ir_datasets.load_tira("apples")           # TIRA-format directory
```

Aliases take precedence over paths. `load_tira` also accepts plain paths. An id that is neither an alias nor an existing directory raises an error that lists the known aliases (`aliases.resolve_location(dataset, fail_if_not_found=True)`). The tests patch the dictionary temporarily.

## PyTerrier

`load_pyterrier_dataset` loads a dataset with `ir_datasets.v2` and returns the matching `pt.datasets.get_dataset(...)` object. It accepts an alias or any v2 id such as `hf-local:<path>[/fragment]`:

```python
from local_ir_datasets import load_pyterrier_dataset

dataset = load_pyterrier_dataset("pangrams")   # alias, or "hf-local:<path>"
topics = dataset.get_topics()                  # qid, query
qrels = dataset.get_qrels()                    # qid, docno, label
docs = dataset.get_corpus_iter()               # dicts with docno, text
```

## Development

Use the dev container (see `AGENTS.md`); it includes Java, PyTerrier, `ir_measures` and `tira`, so the PyTerrier tests run there. Run `pytest`.
