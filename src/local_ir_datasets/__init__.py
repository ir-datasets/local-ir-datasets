"""Local ir_datasets providers.

Importing this package registers the ``hf-local:`` provider with
``ir_datasets.v2`` (see :mod:`local_ir_datasets.hf_local_provider`) and
exposes :func:`local_hf_repo` for temporarily redirecting an ``hf:`` repo id
to a local directory.
"""
from . import hf_local_provider as _hf_local_provider
from .hf_local import local_hf_repo
from .tira_local import load_tira

# Idempotent; no-op if ir_datasets isn't installed.
_hf_local_provider.register()

__all__ = ["local_hf_repo", "load_tira"]
