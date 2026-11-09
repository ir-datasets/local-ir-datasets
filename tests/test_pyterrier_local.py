"""PyTerrier indexing and retrieval over the local datasets (needs Java and
``python-terrier``, both available in the dev container)."""

import tempfile
import unittest
from pathlib import Path

try:
    import pyterrier as pt

    PYTERRIER_AVAILABLE = True
except ImportError:
    PYTERRIER_AVAILABLE = False

from local_ir_datasets import load_tira

TIRA_DATASET_DIR = Path(__file__).parent / "resources" / "example-tira-dataset"


@unittest.skipUnless(PYTERRIER_AVAILABLE, "python-terrier is required (available in the dev container)")
class PyTerrierOnLocalDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not pt.java.started():
            pt.java.init()
        cls.dataset = load_tira(str(TIRA_DATASET_DIR))

    def test_index_and_retrieve_with_bm25(self) -> None:
        with tempfile.TemporaryDirectory() as index_dir:
            docs = ({"docno": d.doc_id, "text": d.text} for d in self.dataset.docs_iter())
            index_ref = pt.IterDictIndexer(index_dir, meta={"docno": 64, "text": 4096}).index(docs)
            index = pt.IndexFactory.of(index_ref)
            self.assertEqual(index.getCollectionStatistics().getNumberOfDocuments(), 8)

            queries = [
                {"qid": "300", "query": "apple"},
                {"qid": "700", "query": "fox"},
            ]
            import pandas as pd

            results = pt.terrier.Retriever(index, wmodel="BM25")(pd.DataFrame(queries))

        self.assertEqual(set(results["qid"]), {"300", "700"})
        self.assertTrue((results["score"] > 0).all())

    def test_evaluate_run_against_qrels_with_ir_measures(self) -> None:
        import ir_measures
        from ir_measures import nDCG

        qrels = [q for q in self.dataset.qrels_iter() if q.relevance > 0]
        perfect_run = [
            ir_measures.ScoredDoc(q.query_id, q.doc_id, float(q.relevance)) for q in qrels
        ]
        result = ir_measures.calc_aggregate([nDCG @ 10], qrels, perfect_run)
        self.assertAlmostEqual(result[nDCG @ 10], 1.0)
