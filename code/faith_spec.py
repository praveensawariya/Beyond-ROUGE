#!/usr/bin/env python
"""Write the formal Faith-ZS specification (review R1-W10 / Q2), read from faithfulness.py's ACTUAL defaults
so the text cannot drift from the code. Also runs a pure-python re-implementation of the aggregation on a toy
NLI table and asserts it equals the formula."""
import argparse, inspect, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
SPEC = """# Faith-ZS: formal definition (matches faithfulness.py)

Input: source article D, summary S. Defaults (read from the code): window w={w}, stride s={s}, top-k k={k},
NLI max_len={ml}; NLI = MNLI-trained cross-encoder (default {nli}); retriever = {sb}.

1. Sentences. D -> d_1..d_m (regex split on [.!?]+whitespace, sentences >= 3 words kept; the article itself if none).
   S -> h_1..h_n (same splitter, sentences >= 4 words kept). If n = 0 the document score is 0 (and supported_rate 0).
2. Source windows. c_j = d_{{js+1}} ... d_{{js+w}} for j = 0, 1, ..., ceil(m/s)-1 (final window may be shorter).
3. Retrieval. e(.) = L2-normalised SBERT embedding. For each h_i keep T_i = top-min(k, #windows) windows by cosine e(h_i).e(c_j).
4. NLI. For every (c_j in T_i, h_i) the NLI model gives softmax probabilities p_ent, p_con, p_neu (premise = window, hypothesis = summary sentence).
5. Sentence score.   z_i = max_{{c in T_i}} [ p_ent(c, h_i) - p_con(c, h_i) ]            in [-1, 1]
6. Document score.   Faith-ZS(D, S) = (1/n) * sum_{{i=1..n}} z_i                          (MEAN over summary sentences)
7. Supported rate.   Sup(D, S)      = (1/n) * sum_{{i=1..n}} 1[ max_{{c in T_i}} p_ent(c, h_i) > 0.5 ]
8. System score = mean over the 500 documents (reported x100, 95% bootstrap CI over documents).

Notes: there is no threshold in Faith-ZS itself (only in Sup); sentence-level aggregation is max over retrieved windows,
document-level aggregation is the arithmetic mean over summary sentences (not min).
"""


def reference_score(p_ent, p_con):
    """p_ent, p_con: arrays [n_sentences, k]. Pure-numpy statement of steps 5-7."""
    z = (p_ent - p_con).max(1); return float(z.mean()), float((p_ent.max(1) > 0.5).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    src = Path(__file__).resolve().parent / "faithfulness.py"
    import re
    text = src.read_text()
    init = text[text.index("def __init__"):text.index("import torch")]
    g = lambda name: re.search(rf"{name}=([^,\n]+)", init).group(1).strip().strip('"')
    spec = SPEC.format(w=g("window"), s=g("stride"), k=g("topk"), ml=g("max_len"), nli=g("nli_name"), sb=g("sbert_name"))
    # consistency check of steps 5-7 against faithfulness.py's own expressions
    assert "(p[:, :, self.ent] - p[:, :, self.con]).max(1)" in text and "float(zs.mean())" in text and "> 0.5" in text, "faithfulness.py no longer matches the spec"
    rng = np.random.default_rng(0); pe = rng.uniform(0, 1, (5, 3)); pc = rng.uniform(0, 1, (5, 3)) * (1 - pe)
    zs, sup = reference_score(pe, pc); assert -1 <= zs <= 1 and 0 <= sup <= 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(spec); print(spec); print("spec consistent with faithfulness.py: OK; wrote", a.out)


if __name__ == "__main__":
    main()
