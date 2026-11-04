"""
Hybrid summarization pipeline: SBERT + MMR extraction -> adaptive context
expansion -> LED abstractive refinement.  Library module (no side effects).
"""
from __future__ import annotations

import gc
import re
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
_PLACEHOLDER = re.compile(r"@x(?:math|cite)\d*")   # ccdv/arxiv-summarization masks
_S_TAGS = re.compile(r"</?S>")                     # sentence tags in some abstracts


def clean_text(text: str) -> str:
    """Remove dataset placeholders and undo word-level tokenisation
    ("short - term ." -> "short-term.").  Applied identically to articles and
    reference abstracts so that every system is scored on the same text."""
    text = _S_TAGS.sub(" ", text)
    text = _PLACEHOLDER.sub(" ", text)
    text = re.sub(r"\s+([.,;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\(\s*\)|\[\s*\]", "", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r"(?<=\w)\s+-\s+(?=\w)", "-", text)
    text = re.sub(r"\s*_\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(article: str) -> list[str]:
    """The dataset stores one sentence per line; fall back to NLTK otherwise."""
    lines = [l for l in article.split("\n") if l.strip()]
    if len(lines) < 5:
        from nltk.tokenize import sent_tokenize
        try:
            lines = sent_tokenize(" ".join(lines))
        except LookupError:
            import nltk
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
            lines = sent_tokenize(" ".join(lines))
    sents = [clean_text(l) for l in lines]
    return [s for s in sents if len(s.split()) >= 3]


# --------------------------------------------------------------------------
# Extractive stage
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractConfig:
    target_words: int = 220        # words in the selected ("core") sentences
    min_k: int = 6                 # adaptive number of sentences:
    max_k: int = 15                #   k = clip(round(0.05 * n_sent), min_k, max_k)
    max_sentences: int | None = None   # overrides the adaptive k if set
    lambda_rel: float = 0.75       # MMR: weight on relevance (1-lambda on redundancy)
    lead_bias: float = 0.12
    use_mmr: bool = True           # False -> plain top-k by score (+trigram blocking)
    window: int = 1                # neighbours added on each side of a selected sentence
    max_input_words: int = 900     # budget for the expanded text fed to the generator
    min_words: int = 6
    max_words: int = 45
    trigram_overlap: float = 0.6
    max_doc_sentences: int = 4000  # safety cap for the n x n similarity matrix


def embed_sentences(sents, sbert, batch_size: int = 64) -> np.ndarray:
    return sbert.encode(sents, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


def _trigrams(text: str):
    t = text.lower().split()
    return set(zip(t, t[1:], t[2:]))


def _too_symbolic(s: str, max_ratio: float = 0.35) -> bool:
    if len(s) < 10:
        return True
    sym = sum(1 for c in s if not c.isalnum() and c not in " .,:;!?()-")
    return sym / len(s) > max_ratio


def rank_sentences(sents, emb, cfg: ExtractConfig) -> list[int]:
    """Return selected sentence ids in selection order (best first)."""
    n = len(sents)
    wc = np.array([len(s.split()) for s in sents])
    centroid = emb.mean(0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    sem = emb @ centroid
    pos = np.linspace(1.0, 0.0, n) if n > 1 else np.ones(1)
    base = sem + cfg.lead_bias * pos
    base = base - 0.10 * np.clip((wc - 25) / 25, 0, 1)

    valid = (wc >= cfg.min_words) & (wc <= cfg.max_words)
    valid &= np.array([not _too_symbolic(s) for s in sents])
    if valid.sum() == 0:
        valid = wc >= 3
    base = np.where(valid, base, -1e9)

    k = cfg.max_sentences or int(np.clip(round(0.05 * n), cfg.min_k, cfg.max_k))
    sim = emb @ emb.T if cfg.use_mmr else None
    max_sim = np.zeros(n)
    avail = valid.copy()
    ranked, tri, words = [], [], 0

    while len(ranked) < k and words < cfg.target_words and avail.any():
        if cfg.use_mmr and ranked:
            score = cfg.lambda_rel * base - (1 - cfg.lambda_rel) * max_sim
        else:
            score = base.copy()
        score = np.where(avail, score, -1e9)
        i = int(np.argmax(score))
        if score[i] < -1e8:
            break
        avail[i] = False
        t = _trigrams(sents[i])
        if t and any(len(t & u) / len(t) >= cfg.trigram_overlap for u in tri):
            continue                                   # redundancy blocking
        ranked.append(i)
        tri.append(t)
        words += int(wc[i])
        if sim is not None:
            max_sim = np.maximum(max_sim, sim[i])
    return ranked


def build_extract(sents, ranked, cfg: ExtractConfig) -> dict:
    """core = selected sentences; expanded = core + neighbouring sentences,
    added in selection order until the word budget is reached."""
    n = len(sents)
    expanded, words = set(), 0
    for i in ranked:
        grp = [j for j in range(max(0, i - cfg.window), min(n, i + cfg.window + 1))
               if j not in expanded]
        add = sum(len(sents[j].split()) for j in grp)
        if expanded and words + add > cfg.max_input_words:
            break
        expanded.update(grp)
        words += add
    core_ids, exp_ids = sorted(ranked), sorted(expanded)
    return {
        "core_ids": core_ids,
        "expanded_ids": exp_ids,
        "core_text": " ".join(sents[j] for j in core_ids),
        "expanded_text": " ".join(sents[j] for j in exp_ids),
    }


def extract(sents, emb, cfg: ExtractConfig) -> dict:
    return build_extract(sents, rank_sentences(sents, emb, cfg), cfg)


# --------------------------------------------------------------------------
# Abstractive stage
# --------------------------------------------------------------------------
def make_gen_kwargs(num_beams: int = 4) -> dict:
    """Identical decoding settings for every neural system (fair comparison)."""
    return dict(num_beams=num_beams, max_new_tokens=300, min_new_tokens=100,
                no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True)


def set_seed(seed: int = 42) -> None:
    """Seed every RNG (reproducibility; beam search itself is deterministic)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def free_gpu():
    """Hand GPU memory back (call after a stage is finished and its objects are deleted)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _sinusoid(n_pos, dim):
    import numpy as np, torch
    pos = np.arange(n_pos)[:, None]; j = np.arange(dim)[None, :]
    pe = pos / np.power(10000, 2 * (j // 2) / dim)
    out = np.zeros((n_pos, dim), dtype=np.float32); sent = dim // 2 if dim % 2 == 0 else dim // 2 + 1
    out[:, :sent] = np.sin(pe[:, 0::2]); out[:, sent:] = np.cos(pe[:, 1::2])
    return torch.from_numpy(out)


def fix_pegasus_positions(model):
    """PEGASUS uses FIXED sinusoidal positions that are not stored in the checkpoint ('newly initialized' warning).
    With low_cpu_mem_usage / fp16 they can end up as garbage -> degenerate summaries. Recompute and verify."""
    import torch
    fixed = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "PegasusSinusoidalPositionalEmbedding" and getattr(mod, "weight", None) is not None:
            w = mod.weight; ww = w.detach().float().cpu(); exp = _sinusoid(*w.shape); half = ww.shape[1] // 2
            # a correct table has position 0 = [0,...,0 | 1,...,1] and finite values; anything else is uninitialised/garbage
            ok = bool(torch.isfinite(ww).all()) and float(ww[0, :half].abs().max()) < 1e-3 and float((ww[0, half:] - 1).abs().max()) < 1e-3
            print(f"[pegasus] {name}: valid={ok}, max|diff to analytic sinusoid|={float((ww - exp).abs().max()):.4f}")
            if not ok:
                with torch.no_grad():
                    w.copy_(exp.to(device=w.device, dtype=w.dtype))
                fixed += 1
    print(f"[pegasus] sinusoidal position tables: {'REPAIRED ' + str(fixed) if fixed else 'already correct'}")


class Seq2Seq:
    """Thin wrapper around a HF seq2seq model (kind in {'led','bart','t5'})."""

    def __init__(self, hf_name: str, kind: str, device: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch, self.kind, self.device = torch, kind, device
        self.tok = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_name, low_cpu_mem_usage=(kind != "pegasus"))
        if kind == "pegasus":
            fix_pegasus_positions(model)
        # fp16 halves VRAM and speeds up the T4; T5 overflows to NaN in fp16 so it stays fp32
        # (t5-base is small). .half() works on every transformers version (torch_dtype= is deprecated).
        if device == "cuda" and kind not in ("t5", "pegasus"):   # PEGASUS stays fp32 (fp16 is unreliable)
            model = model.half()
        self.model = model.to(device).eval()
        from hyb_gpu import assert_on_device, gpu_report
        assert_on_device(self.model, device, f"{kind} model")     # proves the weights are on the GPU
        gpu_report(f"after loading {hf_name}")

    def _generate(self, texts, max_in_tokens, gen):
        torch = self.torch
        if self.kind == "t5":
            texts = ["summarize: " + t for t in texts]
        enc = self.tok(texts, truncation=True, max_length=max_in_tokens, padding=True,
                       return_tensors="pt").to(self.device)
        kw = dict(gen)
        if self.kind == "led":                       # LED requires global attention on the first token
            g = torch.zeros_like(enc["input_ids"])
            g[:, 0] = 1
            kw["global_attention_mask"] = g
        with torch.inference_mode():
            out = self.model.generate(**enc, **kw)
        return [s.replace("<n>", " ").strip() if self.kind == "pegasus" else s.strip()
                for s in self.tok.batch_decode(out, skip_special_tokens=True)]

    def summarize_batch(self, texts, max_in_tokens: int, gen: dict):
        """Several documents per GPU call. On CUDA out-of-memory the batch is retried one by one."""
        try:
            return self._generate(texts, max_in_tokens, gen)
        except self.torch.cuda.OutOfMemoryError:
            free_gpu()
            if len(texts) == 1:
                raise
            print(f"  (GPU out of memory with batch of {len(texts)}; retrying one by one)")
            return [self._generate([t], max_in_tokens, gen)[0] for t in texts]

    def summarize(self, text: str, max_in_tokens: int, gen: dict) -> str:
        return self.summarize_batch([text], max_in_tokens, gen)[0]

    def close(self):
        del self.model, self.tok
        free_gpu()
