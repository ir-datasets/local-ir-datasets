"""Expose local datasets to PyTerrier.

:func:`load_pyterrier_dataset` loads a dataset with ``ir_datasets.v2`` and
returns the matching ``pt.datasets.get_dataset(...)`` object, so
``get_topics()``, ``get_qrels()``, ``get_corpus_iter()`` etc. work on it.
``pyterrier`` is imported lazily.
"""
import hashlib
from pathlib import Path

from .aliases import resolve_alias


def _v2_id(dataset: str) -> str:
    """Normalize ``dataset`` to a stable ``ir_datasets.v2`` id: an alias
    (``<alias>[/fragment]``) becomes ``hf-local:<absolute location>[/fragment]``;
    anything else is passed through unchanged (``hf-local:...``, ``irds:...``)."""
    aliased = resolve_alias(dataset)
    if aliased is None:
        return dataset
    location, fragment = aliased
    spec = str(Path(location).resolve())
    return f"hf-local:{spec}/{fragment}" if fragment else f"hf-local:{spec}"


def load_pyterrier_dataset(dataset: str):
    """Load ``dataset`` -- a dataset alias (see :mod:`local_ir_datasets.aliases`),
    or any ``ir_datasets.v2`` id such as ``hf-local:<path>[/fragment]`` -- and
    return it as a PyTerrier dataset (``pt.datasets.get_dataset``).

    Raises ``KeyError`` (listing the known aliases for ``hf-local:`` ids) if
    the dataset can't be found."""
    import ir_datasets
    import ir_datasets.v2
    import pyterrier as pt

    from . import hf_local_provider

    hf_local_provider.register()
    v2_id = _v2_id(dataset)
    node = ir_datasets.v2.load(v2_id)

    # PyTerrier resolves "irds:<name>" through ir_datasets' (v1) registry,
    # where v2 nodes work too (they keep the legacy method API); the name is
    # derived from the resolved id so changed aliases never reuse stale entries.
    name = f"local-ir-datasets/{hashlib.sha256(v2_id.encode('utf-8')).hexdigest()[:24]}"
    if name not in ir_datasets.registry:
        ir_datasets.registry.register(name, node)
    return pt.datasets.get_dataset(f"irds:{name}")
