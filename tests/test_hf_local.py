import unittest
from pathlib import Path

try:
    import ir_datasets.v2  # noqa: F401
    from ir_datasets.v2 import hf_provider as hfm

    IR_DATASETS_V2_AVAILABLE = True
except ImportError:
    IR_DATASETS_V2_AVAILABLE = False

from local_ir_datasets import local_hf_repo
from local_ir_datasets.hf_local import _local_dirs

DATASET_DIR = Path(__file__).parent / "resources" / "example-hf-dataset"
REPO = "local-ir-datasets/example-hf-dataset-unit-test"


@unittest.skipUnless(
    IR_DATASETS_V2_AVAILABLE, "ir_datasets.v2 is required (available in the dev container)"
)
class LocalHfRepoTest(unittest.TestCase):
    def test_forgets_the_repo_and_clears_caches_on_exit(self) -> None:
        # The hub proxy (see hf_local.py) is installed once, permanently,
        # for the whole process -- possibly by other code (e.g. the
        # `hf-local:` provider) well before this test runs -- so what's
        # actually under test is that `REPO` is only resolvable (and only
        # registered in `_local_dirs`) for the duration of the `with` block,
        # not that `hf_provider._hf_lib` itself changes identity.
        with local_hf_repo(REPO, DATASET_DIR):
            self.assertIn(REPO, _local_dirs)
            ir_datasets.v2.load(f"hf:{REPO}")
            self.assertTrue(any(key[0] == REPO for key in hfm._card_cache))

        self.assertNotIn(REPO, _local_dirs)
        self.assertFalse(any(key[0] == REPO for key in hfm._card_cache))
        self.assertFalse(any(key == REPO for key in hfm._repo_cache))
        self.assertFalse(any(key[0] == REPO for key in hfm._node_cache))

    def test_forgets_the_repo_even_if_the_body_raises(self) -> None:
        with self.assertRaises(ValueError):
            with local_hf_repo(REPO, DATASET_DIR):
                raise ValueError("boom")

        self.assertNotIn(REPO, _local_dirs)

    def test_resolves_documents_from_the_local_directory(self) -> None:
        with local_hf_repo(REPO, DATASET_DIR):
            benchmark = ir_datasets.v2.load(f"hf:{REPO}")
            self.assertEqual(len(list(benchmark.docs)), 10)
