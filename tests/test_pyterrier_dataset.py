"""``load_pyterrier_dataset``: local datasets as PyTerrier datasets."""

import unittest
from pathlib import Path
from unittest import mock

try:
    import pyterrier as pt

    PYTERRIER_AVAILABLE = True
except ImportError:
    PYTERRIER_AVAILABLE = False

from local_ir_datasets import load_pyterrier_dataset
from local_ir_datasets.aliases import DATASET_ALIASES

DATASET_DIR = Path(__file__).parent / "resources" / "example-hf-dataset"
ALIAS = "pangrams"


@unittest.skipUnless(PYTERRIER_AVAILABLE, "python-terrier is required (available in the dev container)")
class LoadPyTerrierDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(DATASET_ALIASES, {ALIAS: str(DATASET_DIR)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _check(self, dataset) -> None:
        topics = dataset.get_topics()
        self.assertEqual(list(topics["qid"]), ["q1", "q2", "q3"])
        self.assertIn("query", topics.columns)

        qrels = dataset.get_qrels()
        self.assertEqual(set(qrels.columns), {"qid", "docno", "label"})
        self.assertEqual(len(qrels), 3)

        docs = list(dataset.get_corpus_iter(verbose=False))
        self.assertEqual(len(docs), 10)
        self.assertEqual({d["docno"] for d in docs}, {f"d{i}" for i in range(1, 11)})
        self.assertIn("text", docs[0])

    def test_loads_by_alias(self) -> None:
        self._check(load_pyterrier_dataset(ALIAS))

    def test_loads_by_hf_local_path(self) -> None:
        self._check(load_pyterrier_dataset(f"hf-local:{DATASET_DIR}"))

    def test_loads_by_hf_local_alias(self) -> None:
        self._check(load_pyterrier_dataset(f"hf-local:{ALIAS}"))

    def test_alias_and_path_load_the_same_dataset(self) -> None:
        by_alias = load_pyterrier_dataset(ALIAS)
        by_path = load_pyterrier_dataset(f"hf-local:{DATASET_DIR}")
        self.assertEqual(repr(by_alias), repr(by_path))

    def test_is_a_pyterrier_dataset(self) -> None:
        self.assertIsInstance(load_pyterrier_dataset(ALIAS), pt.datasets.Dataset)

    def test_unknown_dataset_raises_with_known_aliases(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            load_pyterrier_dataset("hf-local:no-such-alias")
        message = str(ctx.exception)
        self.assertIn("not a known dataset alias", message)
        self.assertIn(ALIAS, message)

    def test_can_index_and_retrieve(self) -> None:
        import tempfile

        import pandas as pd

        if not pt.java.started():
            pt.java.init()
        dataset = load_pyterrier_dataset(ALIAS)
        with tempfile.TemporaryDirectory() as index_dir:
            index_ref = pt.IterDictIndexer(index_dir, meta={"docno": 64, "text": 4096}).index(
                dataset.get_corpus_iter(verbose=False)
            )
            results = pt.terrier.Retriever(pt.IndexFactory.of(index_ref), wmodel="BM25")(
                pd.DataFrame(dataset.get_topics())
            )
        self.assertTrue({"q1", "q2"} <= set(results["qid"]))
