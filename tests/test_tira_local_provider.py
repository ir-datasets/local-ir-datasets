"""Tests for the ``tira-local:`` provider (see
:mod:`local_ir_datasets.tira_local_provider`), addressed by path and by alias,
against ``tests/resources/example-tira-dataset``."""

import unittest
from pathlib import Path
from unittest import mock

try:
    import ir_datasets.v2  # noqa: F401

    IR_DATASETS_V2_AVAILABLE = True
except ImportError:
    IR_DATASETS_V2_AVAILABLE = False

import local_ir_datasets  # noqa: F401  (registers "tira-local:")
from local_ir_datasets.aliases import DATASET_ALIASES
from local_ir_datasets.hf_local_provider import _split_path_and_fragment

DATASET_DIR = Path(__file__).parent / "resources" / "example-tira-dataset"
ALIAS = "apples"
FIRST_DOC_ID = "a6551a7a-505c-4b91-84eb-c85100ea6451"


class SplitPathAndFragmentTest(unittest.TestCase):
    def split(self, spec):
        return _split_path_and_fragment(spec, marker="queries.jsonl", scheme="tira-local")

    def test_bare_directory_has_no_fragment(self) -> None:
        self.assertEqual(self.split(str(DATASET_DIR)), (DATASET_DIR.resolve(), None))

    def test_trailing_segment_is_split_off_as_fragment(self) -> None:
        self.assertEqual(self.split(f"{DATASET_DIR}/docs"), (DATASET_DIR.resolve(), "docs"))

    def test_error_names_the_scheme_and_marker(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            self.split("does-not-exist/docs")
        self.assertIn("tira-local:does-not-exist/docs", str(ctx.exception))
        self.assertIn("queries.jsonl", str(ctx.exception))


@unittest.skipUnless(
    IR_DATASETS_V2_AVAILABLE, "ir_datasets.v2 is required (available in the dev container)"
)
class TiraLocalProviderTest(unittest.TestCase):
    """Run with the directory path, then again with an alias."""

    def spec(self) -> str:
        return str(DATASET_DIR)

    def test_load_bare_directory_resolves_to_the_benchmark(self) -> None:
        benchmark = ir_datasets.v2.load(f"tira-local:{self.spec()}")
        docs = list(benchmark.docs)
        self.assertEqual(len(docs), 8)
        self.assertEqual(docs[0].doc_id, FIRST_DOC_ID)
        self.assertEqual(docs[0].text, "one apple a day keeps doctor away")

        queries = {q.query_id: q.text for q in benchmark.queries}
        self.assertEqual(
            queries,
            {
                "300": "benefits one apple a day",
                "301": "apple",
                "700": "fox jumps",
                "701": "jumping animals",
            },
        )

        qrels = list(benchmark.qrels)
        self.assertEqual(len(qrels), 11)
        self.assertEqual(
            (qrels[0].query_id, qrels[0].doc_id, qrels[0].relevance), ("300", FIRST_DOC_ID, 1)
        )

    def test_fragments_select_single_tables(self) -> None:
        for fragment, expected in (("docs", 8), ("queries", 4), ("qrels", 11)):
            table = ir_datasets.v2.load(f"tira-local:{self.spec()}/{fragment}")
            self.assertEqual(len(list(table)), expected)

    def test_repeated_resolution_shares_one_table(self) -> None:
        first = ir_datasets.v2.load(f"tira-local:{self.spec()}/docs")
        second = ir_datasets.v2.load(f"tira-local:{self.spec()}/docs")
        self.assertIs(first, second)

    def test_docs_store_looks_up_a_document_by_id(self) -> None:
        docs = ir_datasets.v2.load(f"tira-local:{self.spec()}/docs")
        self.assertEqual(docs.docs_store().get(FIRST_DOC_ID).text, "one apple a day keeps doctor away")

    def test_unknown_dataset_raises_with_known_aliases(self) -> None:
        with mock.patch.dict(DATASET_ALIASES, {"other": str(DATASET_DIR)}):
            with self.assertRaises(KeyError) as ctx:
                ir_datasets.v2.load("tira-local:no-such-alias")
        self.assertIn("not a known dataset alias", str(ctx.exception))
        self.assertIn("other", str(ctx.exception))


class TiraLocalProviderAliasTest(TiraLocalProviderTest):
    def spec(self) -> str:
        return ALIAS

    def setUp(self) -> None:
        patcher = mock.patch.dict(DATASET_ALIASES, {ALIAS: str(DATASET_DIR)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_alias_and_path_resolve_to_the_same_table(self) -> None:
        self.assertIs(
            ir_datasets.v2.load(f"tira-local:{ALIAS}/docs"),
            ir_datasets.v2.load(f"tira-local:{DATASET_DIR}/docs"),
        )
