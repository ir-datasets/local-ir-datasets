"""``hf-local:`` -- a Generator-based ``ir_datasets.v2`` provider mirroring
``hf:``'s own addressing scheme, but resolving its "repo" against a local
directory on disk instead of a real Hugging Face Hub repo.

Lets a persisted, ir_datasets-annotated dataset card (e.g.
``tests/resources/example-hf-dataset``) be loaded directly, with no test
setup at all::

    ir_datasets.v2.load("hf-local:tests/resources/example-hf-dataset")
    ir_datasets.v2.load("hf-local:tests/resources/example-hf-dataset/docs")
    verify("hf-local:tests/resources/example-hf-dataset/docs")

Addressing: ``hf-local:<path>[/fragment]``, where ``<path>`` is a filesystem
path (relative to the current working directory, or absolute) to a directory
laid out exactly like a real ``hf:`` dataset repo (a ``README.md`` with an
``ir_datasets:``/``ir-datasets:`` YAML frontmatter card, plus whatever
``file:`` paths it declares), and the optional ``/fragment`` is a raw
``fragment`` of one of the card's entries -- exactly like ``hf:``'s own fragment. Since ``<path>`` may
itself contain ``/``, it is *not* found by splitting on the first/last ``/``
alone: the whole spec is first tried as a directory outright, and only if
that isn't one is its last path segment peeled off and tried as a fragment
instead (see :func:`_split_path_and_fragment`).

Each local directory is mapped to a synthetic, ``owner/name``-shaped repo id
(see :func:`_repo_id_for`) registered with
:mod:`local_ir_datasets.hf_local`'s shared hub proxy, then resolved
via ``hf_provider``'s own ``_HfResolver`` -- the real card-parsing/table-
building logic runs unmodified; only where it fetches its files (a local
directory instead of a Hub download) differs.

TODO: like :mod:`local_ir_datasets.hf_local` (whose hub proxy this
module reuses and permanently installs), this is a workaround for
``ir_datasets`` having no first-class "local hf: repo" feature of its own;
rewrite/remove this once it does.
"""
import hashlib
from pathlib import Path
from typing import Optional, Tuple

from . import hf_local
from .aliases import known_aliases_hint, resolve_alias


def _repo_id_for(local_dir: Path) -> str:
    """A stable, ``owner/name``-shaped (one slash, no ``@``) synthetic repo
    id for a resolved local directory -- the same directory always maps to
    the same id, so repeated resolutions of the same path share
    ``hf_provider``'s own internal caches instead of rebuilding independent
    objects."""
    digest = hashlib.sha256(str(local_dir).encode("utf-8")).hexdigest()[:24]
    return f"local/{digest}"


def _split_path_and_fragment(
    spec: str, marker: str = "README.md", scheme: str = "hf-local"
) -> Tuple[Path, Optional[str]]:
    """``<path>[/fragment]`` -> ``(resolved directory, fragment or None)``.

    Tries ``spec`` as ``<alias>[/fragment]`` (see
    :mod:`local_ir_datasets.aliases`) first, then the whole ``spec`` as a
    directory (containing the ``marker`` file); only if that fails is its last
    ``/``-segment peeled off and tried as a fragment,
    with the remainder re-checked the same way. Raises
    ``KeyError`` if neither works -- becomes "no such node" at
    ``ManifestProvider.__getitem__``, exactly like an unannotated ``hf:``
    repo does. ``marker`` and ``scheme`` let ``tira-local:`` reuse this.
    """
    aliased = resolve_alias(spec)
    if aliased is not None:
        location, fragment = aliased
        if (Path(location) / marker).is_file():
            return Path(location).resolve(), fragment
        raise KeyError(f"{scheme}:{spec}: alias location {location} has no {marker}")
    candidate = Path(spec)
    if (candidate / marker).is_file():
        return candidate.resolve(), None
    if "/" in spec:
        parent, _, fragment = spec.rpartition("/")
        parent_path = Path(parent)
        if (parent_path / marker).is_file():
            return parent_path.resolve(), fragment
    raise KeyError(
        f"{scheme}:{spec}: not a known dataset alias, and neither {candidate} nor a "
        f"fragment split off its last path segment is a directory containing a {marker}. "
        f"{known_aliases_hint()}"
    )


def _resolve(spec: str):
    from ir_datasets.v2 import hf_provider as hfm

    local_dir, fragment = _split_path_and_fragment(spec)
    hf_local._ensure_installed()
    repo = _repo_id_for(local_dir)
    hf_local._local_dirs[repo] = local_dir
    hf_spec = f"{repo}/{fragment}" if fragment else repo
    return hfm._HfResolver(hf_spec)


provider = None
try:
    from ir_datasets.v2.base import Generator, Param
    from ir_datasets.v2.nodes import TABLE
    from ir_datasets.v2.registry import ManifestProvider
except ImportError:  # ir_datasets is optional; the provider is then simply unavailable
    pass
else:
    provider = ManifestProvider("hf-local")
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


def register() -> None:
    """Add the ``hf-local`` provider to ``ir_datasets.v2``'s default graph.
    Installed packages get this through the ``ir_datasets.providers`` entry
    point; calling it is only needed when running from a source tree.
    Idempotent; does nothing if ``ir_datasets`` isn't installed."""
    if provider is None:
        return
    from ir_datasets.v2.graph import default_graph
    default_graph().add(provider)
