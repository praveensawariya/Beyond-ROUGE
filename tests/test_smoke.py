"""CPU-only smoke tests. No model is downloaded and no GPU is needed.

Run from the repository root:  python -m unittest discover -s tests -v
or simply:  python tests/test_smoke.py
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np

from faithfulness import sent_split
from pipeline import ExtractConfig, build_extract, clean_text, rank_sentences, split_sentences
from repair import NUM


class TestPipeline(unittest.TestCase):
    def test_clean_text(self):
        out = clean_text("The short - term effect @xmath12 is shown .")
        self.assertIn("short-term", out)
        self.assertNotIn("@xmath", out)
        self.assertNotIn(" .", out)

    def test_split_sentences(self):
        article = "\n".join([
            "The first sentence of this paper is here.",
            "The second sentence of this paper is here.",
            "The third sentence of this paper is here.",
            "The fourth sentence of this paper is here.",
            "The fifth sentence of this paper is here.",
            "The sixth sentence of this paper is here.",
        ])
        sents = split_sentences(article)
        self.assertGreaterEqual(len(sents), 5)
        self.assertTrue(all(len(s.split()) >= 3 for s in sents))

    def test_extract_with_fake_embeddings(self):
        rng = np.random.default_rng(0)
        sents = [f"Sentence number {i} about the method and the results of the study."
                 for i in range(20)]
        emb = rng.normal(size=(len(sents), 16))
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        cfg = ExtractConfig(target_words=180, lambda_rel=0.75, lead_bias=0.2)
        ranked = rank_sentences(sents, emb, cfg)
        self.assertTrue(len(ranked) > 0)
        out = build_extract(sents, ranked, cfg)
        self.assertTrue(set(out["core_ids"]) <= set(out["expanded_ids"]))
        self.assertTrue(len(out["expanded_text"]) >= len(out["core_text"]))

    def test_tuned_configs_match_paper(self):
        arxiv = json.loads((ROOT / "configs" / "best_extract_arxiv.json").read_text())["best"]
        pubmed = json.loads((ROOT / "configs" / "best_extract_pubmed.json").read_text())["best"]
        self.assertEqual((arxiv["lambda_rel"], arxiv["lead_bias"], arxiv["target_words"]),
                         (0.75, 0.2, 180))
        self.assertEqual((pubmed["lambda_rel"], pubmed["lead_bias"], pubmed["target_words"]),
                         (0.9, 0.06, 180))


class TestFaithfulnessAndRepair(unittest.TestCase):
    def test_sent_split(self):
        text = ("The first sentence of the summary is here. "
                "The second sentence of the summary is here. "
                "The third sentence of the summary is here?")
        self.assertEqual(len(sent_split(text)), 3)

    def test_number_regex(self):
        self.assertEqual(NUM.findall("There are 42 papers and 3.5 percent."), ["42", "3.5"])


if __name__ == "__main__":
    unittest.main()
