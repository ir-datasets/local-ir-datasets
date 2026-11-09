"""Load a local TIRA-format dataset directory (``queries.jsonl``,
``corpus.jsonl.gz``/``documents.jsonl[.gz]``, optional ``qrels.txt``) via
``tira``'s ir_datasets integration, addressed by path or by an alias from
:mod:`local_ir_datasets.aliases`.
"""
from .aliases import resolve_location


def load_tira(dataset: str):
    """Load ``dataset`` (an alias from ``DATASET_ALIASES``, or a directory
    path) with ``tira``'s ir_datasets integration. ``tira`` is imported
    lazily, so this module is importable without it. Raises
    ``FileNotFoundError`` (listing the known aliases) if ``dataset`` is
    neither a known alias nor an existing directory."""
    location = resolve_location(dataset, fail_if_not_found=True)

    from tira.third_party_integrations import ir_datasets

    if ir_datasets is None:
        raise ImportError("tira's ir_datasets integration is unavailable (is ir_datasets installed?)")
    return ir_datasets.load(location)
