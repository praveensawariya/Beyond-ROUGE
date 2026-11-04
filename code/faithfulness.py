"""
Retrieval-filtered NLI faithfulness (a SummaC-ZS-style metric, our own implementation).

For every summary sentence:
  1. retrieve the top-k most similar windows of source sentences (SBERT cosine),
  2. score each (window -> summary sentence) pair with an MNLI-trained NLI model,
  3. sentence score = max over windows of P(entail) - P(contradict).
Document score "faith_zs" = mean over summary sentences (range -1..1).
"supported_rate" = share of summary sentences whose best window has P(entail) > 0.5.
"""
from __future__ import annotations

import contextlib
import os
import re

import numpy as np

_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NLI_FP16 = os.environ.get("HYB_NLI_FP16", "1") != "0"   # set HYB_NLI_FP16=0 for FP16-fragile NLI models


def sent_split(text: str, min_words: int = 4) -> list[str]:
    return [
        s.strip()
        for s in _SPLIT.split(text)
        if len(s.split()) >= min_words
    ]


class FaithfulnessScorer:
    def __init__(
        self,
        nli_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        sbert_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        window=3,
        stride=2,
        topk=3,
        batch_size=16,
        max_len=320,
    ):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch, self.device = torch, device
        self.window, self.stride, self.topk = window, stride, topk
        self.batch_size, self.max_len = batch_size, max_len

        self.sbert = SentenceTransformer(sbert_name, device=device)
        self.tok = AutoTokenizer.from_pretrained(nli_name)
        self.nli = (
            AutoModelForSequenceClassification
            .from_pretrained(nli_name)
            .to(device)
            .eval()
        )

        labels = {
            int(k): v.lower()
            for k, v in self.nli.config.id2label.items()
        }
        self.ent = next(
            k for k, v in labels.items() if "entail" in v
        )
        self.con = next(
            k for k, v in labels.items() if "contra" in v
        )
        if not _NLI_FP16:
            self.nli.float()                 # full-precision weights AND no autocast
        self._cache: dict = {}

    def _source(self, doc_id, article):
        if doc_id not in self._cache:
            sents = sent_split(article, min_words=3) or [article]
            chunks = [
                " ".join(sents[i:i + self.window])
                for i in range(0, len(sents), self.stride)
            ]
            emb = self.sbert.encode(
                chunks,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._cache[doc_id] = (chunks, emb)

        return self._cache[doc_id]

    def _nli_probs(self, pairs):
        torch, out = self.torch, []

        for b in range(0, len(pairs), self.batch_size):
            batch = pairs[b:b + self.batch_size]
            if not batch:
                continue

            prem = [str(p or " ") for p, h in batch]
            hyp = [str(h or " ") for p, h in batch]

            # FIX: longest-first truncation avoids the previous
            # "Sequence to truncate too short" tokenizer error.
            enc = self.tok(
                prem,
                hyp,
                truncation=True,
                max_length=self.max_len,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            amp = (
                torch.autocast("cuda", dtype=torch.float16, enabled=_NLI_FP16)
                if str(self.device).startswith("cuda")
                else contextlib.nullcontext()
            )

            with torch.inference_mode(), amp:
                # DEBERTA_V1_FP32_PATCH: v1 DeBERTa overflows in fp16 under transformers 5.x
                import torch as _t
                if getattr(self.nli.config, 'model_type', '') == 'deberta':
                    self.nli.float()
                    with _t.autocast('cuda', enabled=False):
                        logits = self.nli(**enc).logits.float()
                else:
                    logits = self.nli(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1)
                out.append(probs.cpu().numpy())

        if not out:
            return np.empty((0, len(self.nli.config.id2label)))

        return np.concatenate(out, axis=0)

    def score(self, doc_id, article: str, summary: str):
        """Return (faith_zs, supported_rate); empty summaries score (0, 0)."""
        hyps = sent_split(summary)

        if not hyps:
            return 0.0, 0.0

        chunks, cemb = self._source(doc_id, article)
        k = min(self.topk, len(chunks))

        if k == 0:
            return 0.0, 0.0

        hemb = self.sbert.encode(
            hyps,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        top = np.argsort(-(hemb @ cemb.T), axis=1)[:, :k]
        pairs = [
            (chunks[j], h)
            for h, row in zip(hyps, top)
            for j in row
        ]

        p = self._nli_probs(pairs).reshape(len(hyps), k, -1)
        zs = (p[:, :, self.ent] - p[:, :, self.con]).max(1)
        sup = p[:, :, self.ent].max(1) > 0.5

        return float(zs.mean()), float(sup.mean())
