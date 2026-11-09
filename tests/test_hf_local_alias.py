"""The ``hf-local:`` provider tests from ``test_hf_local_provider.py``,
repeated with a short alias (see :mod:`local_ir_datasets.aliases`) instead of
a path."""

import unittest
from pathlib import Path
from unittest import mock

try:
    import ir_datasets.v2  # noqa: F401

    IR_DATASETS_V2_AVAILABLE = True
except ImportError:
    IR_DATASETS_V2_AVAILABLE = False

import local_ir_datasets  # noqa: F401  (registers "hf-local:")
from local_ir_datasets.aliases import DATASET_ALIASES
from local_ir_datasets.hf_local_provider import _split_path_and_fragment

DATASET_DIR = Path(__file__).parent / "resources" / "example-hf-dataset"
ALIAS = "pangrams"


class AliasTestCase(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(DATASET_ALIASES, {ALIAS: str(DATASET_DIR)})
        patcher.start()
        self.addCleanup(patcher.stop)


class AliasesFileTest(unittest.TestCase):
    def test_no_aliases_are_registered_by_default(self) -> None:
        self.assertEqual(DATASET_ALIASES, {})


class SplitAliasAndFragmentTest(AliasTestCase):
    def test_bare_alias_has_no_fragment(self) -> None:
        directory, fragment = _split_path_and_fragment(ALIAS)
        self.assertEqual(directory, DATASET_DIR.resolve())
        self.assertIsNone(fragment)

    def test_trailing_segment_is_split_off_as_fragment(self) -> None:
        directory, fragment = _split_path_and_fragment(f"{ALIAS}/docs")
        self.assertEqual(directory, DATASET_DIR.resolve())
        self.assertEqual(fragment, "docs")

    def test_unknown_alias_raises(self) -> None:
        with self.assertRaises(KeyError):
            _split_path_and_fragment("no-such-alias/docs")

    def test_alias_pointing_to_a_non_dataset_directory_raises(self) -> None:
        with mock.patch.dict(DATASET_ALIASES, {"broken": str(DATASET_DIR.parent / "does-not-exist")}):
            with self.assertRaises(KeyError):
                _split_path_and_fragment("broken/docs")


@unittest.skipUnless(
    IR_DATASETS_V2_AVAILABLE, "ir_datasets.v2 is required (available in the dev container)"
)
class HfLocalProviderAliasTest(AliasTestCase):
    def test_load_bare_alias_resolves_to_the_default_benchmark(self) -> None:
        benchmark = ir_datasets.v2.load(f"hf-local:{ALIAS}")
        self.assertEqual(len(list(benchmark.docs)), 10)

    def test_repeated_resolution_of_the_same_alias_shares_one_table(self) -> None:
        first = ir_datasets.v2.load(f"hf-local:{ALIAS}/docs")
        second = ir_datasets.v2.load(f"hf-local:{ALIAS}/docs")
        self.assertIs(first, second)

    def test_alias_and_path_resolve_to_the_same_table(self) -> None:
        by_alias = ir_datasets.v2.load(f"hf-local:{ALIAS}/docs")
        by_path = ir_datasets.v2.load(f"hf-local:{DATASET_DIR}/docs")
        self.assertIs(by_alias, by_path)

    def test_fragment_selects_a_single_table(self) -> None:
        queries = list(ir_datasets.v2.load(f"hf-local:{ALIAS}/queries"))
        self.assertEqual(len(queries), 3)

    def test_unknown_alias_raises(self) -> None:
        with self.assertRaises(Exception) as ctx:
            ir_datasets.v2.load("hf-local:no-such-alias")
        message = str(ctx.exception)
        self.assertIn("no-such-alias", message)
        self.assertIn("not a known dataset alias", message)
        self.assertIn(ALIAS, message)  # lists the aliases that do exist


@unittest.skipUnless(
    IR_DATASETS_V2_AVAILABLE, "ir_datasets.v2 is required (available in the dev container)"
)
class HfLocalAliasIntegrationTest(AliasTestCase):
    def test_hf_dataset_loads_correctly(self) -> None:
        benchmark = ir_datasets.v2.load(f"hf-local:{ALIAS}")

        docs = list(benchmark.docs)
        queries = list(benchmark.queries)
        qrels = list(benchmark.qrels)
        self.assertEqual({doc.doc_id for doc in docs}, {f"d{i}" for i in range(1, 11)})
        self.assertEqual({qrel.query_id for qrel in qrels}, {"q1", "q2", "q3"})
        self.assertEqual(len(queries), 3)

        for table, expected in (("docs", 10), ("queries", 3), ("qrels", 3)):
            loaded = ir_datasets.v2.load(f"hf-local:{ALIAS}/{table}")
            self.assertEqual(len(list(loaded)), expected)
