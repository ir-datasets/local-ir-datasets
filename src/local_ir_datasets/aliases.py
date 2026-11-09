"""Shortcuts from short dataset ids (aliases) to local dataset locations.

Add entries to :data:`DATASET_ALIASES` to make a dataset addressable by a
short id instead of its path, e.g.::

    DATASET_ALIASES = {"my-corpus": "/data/corpora/my-corpus"}

after which ``hf-local:my-corpus``, ``hf-local:my-corpus/docs`` and
``local_ir_datasets.load_tira("my-corpus")`` all resolve to that directory.
Relative locations are relative to the current working directory.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

#: alias -> location of the dataset directory. Empty by default.
DATASET_ALIASES: Dict[str, str] = {}


def resolve_alias(spec: str) -> Optional[Tuple[str, Optional[str]]]:
    """``<alias>[/fragment]`` -> ``(location, fragment or None)``, or
    ``None`` if the part of ``spec`` before the first ``/`` is not a known
    alias."""
    alias, _, fragment = spec.partition("/")
    if alias not in DATASET_ALIASES:
        return None
    return DATASET_ALIASES[alias], fragment or None


def known_aliases_hint() -> str:
    """Human-readable list of the known aliases and their locations."""
    if not DATASET_ALIASES:
        return "No dataset aliases are registered (see local_ir_datasets.aliases.DATASET_ALIASES)."
    known = ", ".join(f"{alias!r} -> {location!r}" for alias, location in sorted(DATASET_ALIASES.items()))
    return f"Known dataset aliases: {known}."


def resolve_location(dataset: str, fail_if_not_found: bool = True) -> Optional[str]:
    """Location of ``dataset``: its alias target if it is a known alias,
    otherwise ``dataset`` itself if that is an existing directory.

    If neither holds, raises ``FileNotFoundError`` listing the known aliases
    when ``fail_if_not_found`` is true, else returns ``None``."""
    location = DATASET_ALIASES.get(dataset, dataset)
    if Path(location).is_dir():
        return location
    if not fail_if_not_found:
        return None
    if location == dataset:
        problem = f"{dataset!r} is neither a known dataset alias nor an existing directory."
    else:
        problem = f"dataset alias {dataset!r} points to {location!r}, which is not an existing directory."
    raise FileNotFoundError(f"{problem} {known_aliases_hint()}")
