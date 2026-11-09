"""Integration test for the ``hf-local:`` provider (see
:mod:`local_ir_datasets.hf_local_provider`), run against a real,
persisted dataset card (see ``tests/resources/example-hf-dataset``) -- 10
pangram documents, 3 queries, and one relevant document per query.

This uses no mocking, and no explicit local-repo context manager either:
``hf-local:<path>[/fragment]`` is a first-class ``ir_datasets.v2`` provider
that resolves a local directory directly, reusing the real ``hf:`` card-
parsing/row-reading logic end to end against real files on disk, with no
real Hugging Face Hub repo and no network access involved.
"""

import unittest
from pathlib import Path

try:
    import ir_datasets.v2  # noqa: F401

    IR_DATASETS_V2_AVAILABLE = True
except ImportError:
    IR_DATASETS_V2_AVAILABLE = False

import local_ir_datasets  # noqa: F401  (registers "hf-local:")

DATASET_DIR = Path(__file__).parent / "resources" / "example-hf-dataset"


@unittest.skipUnless(
    IR_DATASETS_V2_AVAILABLE, "ir_datasets.v2 is required (available in the dev container)"
)
class HfLocalPangramIntegrationTest(unittest.TestCase):
    def test_hf_dataset_loads_correctly(self) -> None:
        benchmark = ir_datasets.v2.load(f"hf-local:{DATASET_DIR}")

        docs = list(benchmark.docs)
        queries = list(benchmark.queries)
        qrels = list(benchmark.qrels)
        self.assertEqual(len(docs), 10)
        self.assertEqual(len(queries), 3)
        self.assertEqual(len(qrels), 3)
        self.assertEqual({doc.doc_id for doc in docs}, {f"d{i}" for i in range(1, 11)})
        # Exactly one relevant document per query.
        self.assertEqual({qrel.query_id for qrel in qrels}, {"q1", "q2", "q3"})
        self.assertEqual([qrel.relevance for qrel in qrels], [1, 1, 1])

        for table, expected in (("docs", 10), ("queries", 3), ("qrels", 3)):
            loaded = ir_datasets.v2.load(f"hf-local:{DATASET_DIR}/{table}")
            self.assertEqual(len(list(loaded)), expected)
