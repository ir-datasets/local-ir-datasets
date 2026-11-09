"""Tests for loading a local TIRA-format dataset directory
(``tests/resources/example-tira-dataset``: ``corpus.jsonl.gz``,
``queries.jsonl`` and ``qrels.txt``) directly with ``tira``'s ir_datasets
integration, with no network access.
"""

import unittest
from pathlib import Path

try:
    from tira.third_party_integrations import ir_datasets as tira_ir_datasets

    TIRA_AVAILABLE = tira_ir_datasets is not None
except ImportError:
    TIRA_AVAILABLE = False

DATASET_DIR = Path(__file__).parent / "resources" / "example-tira-dataset"


@unittest.skipUnless(TIRA_AVAILABLE, "tira is required (available in the dev container)")
class TiraLocalDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = tira_ir_datasets.load(str(DATASET_DIR))

    def test_dataset_has_docs_queries_and_qrels(self) -> None:
        self.assertTrue(self.dataset.has_docs())
        self.assertTrue(self.dataset.has_queries())
        self.assertTrue(self.dataset.has_qrels())

    def test_docs_are_loaded_from_the_gzipped_corpus(self) -> None:
        docs = list(self.dataset.docs_iter())
        self.assertEqual(len(docs), 8)
        first = docs[0]
        self.assertEqual(first.doc_id, "a6551a7a-505c-4b91-84eb-c85100ea6451")
        self.assertEqual(first.text, "one apple a day keeps doctor away")

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
