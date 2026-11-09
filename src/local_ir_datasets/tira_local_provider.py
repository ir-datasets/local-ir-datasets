"""``tira-local:`` -- a Generator-based ``ir_datasets.v2`` provider for local
TIRA-format dataset directories: ``queries.jsonl`` (``qid``, ``query``), a
corpus (``corpus.jsonl.gz``, ``documents.jsonl.gz`` or ``documents.jsonl``;
``doc_id``, ``default_text``) and an optional TREC ``qrels.txt``::

    ir_datasets.v2.load("tira-local:tests/resources/example-tira-dataset")
    ir_datasets.v2.load("tira-local:tests/resources/example-tira-dataset/docs")

Addressing: ``tira-local:<path-or-alias>[/fragment]``, exactly like
``hf-local:`` (see :mod:`local_ir_datasets.hf_local_provider`), with the
fragments ``docs``, ``queries`` and ``qrels``.

Implementation: the directory is exposed through a generated ``hf:`` dataset
card in a cache directory under the ir_datasets home -- the corpus and queries
are symlinked, ``qrels.txt`` is converted to ``qrels.jsonl`` -- which is then
resolved by ``hf-local:``. So no ``tira`` import or network access is needed.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from . import hf_local_provider

_DOCS_FILES = ("documents.jsonl.gz", "corpus.jsonl.gz", "documents.jsonl")
_MARKER = "queries.jsonl"


def _first_row(path: Path) -> Dict:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return {}


def _link(source: Path, target: Path) -> None:
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(source, target)


def _convert_qrels(source: Path, target: Path) -> None:
    with open(source, encoding="utf-8") as src, open(target, "w", encoding="utf-8") as dst:
        for line in src:
            parts = line.split()
            if not parts:
                continue
            query_id, _iteration, doc_id, relevance = parts[:4]
            dst.write(
                json.dumps({"query_id": query_id, "doc_id": doc_id, "relevance": int(relevance)}) + "\n"
            )


def _column_map(row: Dict, renames: Dict[str, str], drop: List[str]) -> str:
    entries = [f"{src}: {dst}" for src, dst in renames.items() if src in row and dst not in row]
    entries += [f"{col}: null" for col in drop if col in row]
    return "{" + ", ".join(entries) + "}"


def _prepare_card_directory(local_dir: Path) -> Path:
    """Generate (or refresh) the ``hf:``-style directory for ``local_dir``."""
    from ir_datasets.util import home_path

    digest = hashlib.sha256(str(local_dir).encode("utf-8")).hexdigest()[:24]
    card_dir = Path(home_path()) / "local-ir-datasets" / "tira-local" / digest
    card_dir.mkdir(parents=True, exist_ok=True)

    docs_name = next((name for name in _DOCS_FILES if (local_dir / name).is_file()), None)
    if docs_name is None:
        raise KeyError(f"tira-local: {local_dir} has none of {', '.join(_DOCS_FILES)}")
    _link(local_dir / docs_name, card_dir / docs_name)
    _link(local_dir / _MARKER, card_dir / _MARKER)
    has_qrels = (local_dir / "qrels.txt").is_file()
    if has_qrels:
        _convert_qrels(local_dir / "qrels.txt", card_dir / "qrels.jsonl")

    docs_map = _column_map(_first_row(local_dir / docs_name), {"default_text": "text"}, ["segments"])
    queries_map = _column_map(
        _first_row(local_dir / _MARKER), {"qid": "query_id", "query": "text"}, []
    )
    qrels_facet = "    qrels: qrels\n" if has_qrels else ""
    qrels_entry = (
        "  - fragment: qrels\n    type: irds:Qrels\n    file: qrels.jsonl\n" if has_qrels else ""
    )
    card = (
        "---\n"
        "ir_datasets:\n"
        "  - fragment: null\n"
        "    type: irds:AdhocBenchmark\n"
        "    docs: docs\n"
        "    queries: queries\n"
        f"{qrels_facet}"
        "  - fragment: docs\n"
        "    type: irds:Docs\n"
        f"    file: {docs_name}\n"
        f"    column_map: {docs_map}\n"
        "  - fragment: queries\n"
        "    type: irds:Queries\n"
        f"    file: {_MARKER}\n"
        f"    column_map: {queries_map}\n"
        f"{qrels_entry}"
        "---\n"
    )
    (card_dir / "README.md").write_text(card, encoding="utf-8")
    return card_dir


def _resolve(spec: str):
    local_dir, fragment = hf_local_provider._split_path_and_fragment(
        spec, marker=_MARKER, scheme="tira-local"
    )
    card_dir = _prepare_card_directory(local_dir)
    return hf_local_provider._resolve(f"{card_dir}/{fragment}" if fragment else str(card_dir))


_registered = False


def register() -> None:
    """Register the ``tira-local`` provider with ``ir_datasets.v2``'s default
    graph, once per process. Safe to call repeatedly; does nothing if
    ``ir_datasets`` isn't installed."""
    global _registered
    if _registered:
        return
    try:
        from ir_datasets.v2.base import Generator, Param
        from ir_datasets.v2.graph import default_graph
        from ir_datasets.v2.nodes import TABLE
        from ir_datasets.v2.registry import ManifestProvider
    except ImportError:
        return

    provider = ManifestProvider("tira-local")
    provider.register_generator(
        Generator(
            "{spec}",
            params={"spec": Param(pattern=r".+")},
            type=TABLE,
            resolver=_resolve,
            enumerable=False,
        ),
        module=__name__,
    )
    default_graph().add(provider)
    _registered = True


register()
