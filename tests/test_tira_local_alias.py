"""The ``tira`` local dataset tests from ``test_tira_local.py``, repeated
with a short alias (see :mod:`local_ir_datasets.aliases`) loaded via
:func:`local_ir_datasets.load_tira`."""

import unittest
from pathlib import Path
from unittest import mock

try:
    from tira.third_party_integrations import ir_datasets as tira_ir_datasets

    TIRA_AVAILABLE = tira_ir_datasets is not None
except ImportError:
    TIRA_AVAILABLE = False

from local_ir_datasets import load_tira
from local_ir_datasets.aliases import DATASET_ALIASES

DATASET_DIR = Path(__file__).parent / "resources" / "example-tira-dataset"
ALIAS = "apples"


@unittest.skipUnless(TIRA_AVAILABLE, "tira is required (available in the dev container)")
class TiraLocalAliasTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(DATASET_ALIASES, {ALIAS: str(DATASET_DIR)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dataset = load_tira(ALIAS)

    def test_dataset_has_docs_queries_and_qrels(self) -> None:
        self.assertTrue(self.dataset.has_docs())
        self.assertTrue(self.dataset.has_queries())
        self.assertTrue(self.dataset.has_qrels())

    def test_docs_are_loaded_from_the_gzipped_corpus(self) -> None:
        docs = list(self.dataset.docs_iter())
        self.assertEqual(len(docs), 8)
        self.assertEqual(docs[0].doc_id, "a6551a7a-505c-4b91-84eb-c85100ea6451")
        self.assertEqual(docs[0].text, "one apple a day keeps doctor away")

    def test_queries_are_loaded(self) -> None:
        queries = {q.query_id: q.text for q in self.dataset.queries_iter()}
        self.assertEqual(
            queries,
            {
                "300": "benefits one apple a day",
                "301": "apple",
                "700": "fox jumps",
                "701": "jumping animals",
            },
        )

    def test_qrels_are_loaded(self) -> None:
        qrels = list(self.dataset.qrels_iter())
        self.assertEqual(len(qrels), 11)
        self.assertEqual(
            (qrels[0].query_id, qrels[0].doc_id, qrels[0].relevance),
            ("300", "a6551a7a-505c-4b91-84eb-c85100ea6451", 1),
        )

    def test_docs_store_looks_up_a_document_by_id(self) -> None:
        doc = self.dataset.docs_store().get("a6551a7a-505c-4b91-84eb-c85100ea6451")
        self.assertEqual(doc.text, "one apple a day keeps doctor away")

    def test_a_path_still_works_without_an_alias(self) -> None:
        self.assertEqual(len(list(load_tira(str(DATASET_DIR)).docs_iter())), 8)

    def test_unknown_alias_and_missing_directory_raises_with_known_aliases(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            load_tira("no-such-alias")
        message = str(ctx.exception)
        self.assertIn("'no-such-alias' is neither a known dataset alias nor an existing directory", message)
        self.assertIn(f"{ALIAS!r} -> {str(DATASET_DIR)!r}", message)

    def test_alias_pointing_to_a_missing_directory_raises(self) -> None:
        with mock.patch.dict(DATASET_ALIASES, {"broken": "/does/not/exist"}):
            with self.assertRaises(FileNotFoundError) as ctx:
                load_tira("broken")
        self.assertIn("points to '/does/not/exist'", str(ctx.exception))

    def test_missing_dataset_message_says_when_no_aliases_exist(self) -> None:
        with mock.patch.dict(DATASET_ALIASES, clear=True):
            with self.assertRaises(FileNotFoundError) as ctx:
                load_tira("no-such-alias")
        self.assertIn("No dataset aliases are registered", str(ctx.exception))
