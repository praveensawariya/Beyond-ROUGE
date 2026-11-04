"""
Gate 3 self-test — run BEFORE any GPU generation. No models are loaded.

Checks, all against the ORIGINAL pipeline code in this folder:
  1. Each dataset's base config (from its best_extract_*.json, via the original
     load_base_cfg) reproduces the 'full' variant recorded in the results directory's
     run_config.json for the n=100 ablation run — field for field.
  2. Every ablation variant differs from 'full' in EXACTLY ONE field, and
     no_ctx differs ONLY in window (target_words / max_k held constant — the
     Gate 3 fix; the old run changed all three).
  3. Functional check of the window variable itself, on synthetic sentences
     with deterministic embeddings: window=0 -> expanded ids == core ids;
     window=1 -> expanded ids strictly contain the core ids. This proves the
     single field being ablated is the one that drives context expansion in
     pipeline.build_extract.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import ExtractConfig, build_extract, rank_sentences  # noqa: E402
from run_systems import load_base_cfg, make_variants  # noqa: E402

HERE = Path(__file__).parent

# 'full' variants recorded in results/{arxiv,pubmed}/run_config.json
EXPECTED_FULL = {
    "arxiv": dict(target_words=180, min_k=6, max_k=15, max_sentences=None,
                  lambda_rel=0.75, lead_bias=0.2, use_mmr=True, window=1,
                  max_input_words=900, min_words=6, max_words=45,
                  trigram_overlap=0.6, max_doc_sentences=4000),
    "pubmed": dict(target_words=180, min_k=6, max_k=15, max_sentences=None,
                   lambda_rel=0.9, lead_bias=0.06, use_mmr=True, window=1,
                   max_input_words=900, min_words=6, max_words=45,
                   trigram_overlap=0.6, max_doc_sentences=4000),
}
EXPECTED_DIFF = {"no_mmr": "use_mmr", "no_lead": "lead_bias", "no_ctx": "window"}


def main():
    for ds, expected in EXPECTED_FULL.items():
        base = load_base_cfg(str(HERE / f"best_extract_{ds}.json"))
        got = asdict(base)
        assert got == expected, f"{ds} base config mismatch:\n got {got}\n exp {expected}"
        print(f"[selftest] {ds} base config == Drive run_config 'full' variant, field for field")

        variants = make_variants(base)  # also runs assert_single_factor inside
        full = asdict(variants["full"])
        for name, field in EXPECTED_DIFF.items():
            diff = {k for k, v in asdict(variants[name]).items() if v != full[k]}
            assert diff == {field}, f"{ds}/{name}: diff {diff} != {{{field}}}"
        # training configs (variant_cfgs/*.json) must equal the generation configs (make_variants)
        for name in EXPECTED_DIFF:
            trained = asdict(load_base_cfg(str(HERE / "variant_cfgs" / f"{name}_{ds}.json")))
            assert trained == asdict(variants[name]), f"{ds}/{name}: training cfg != generation cfg"
        print(f"[selftest] {ds} training cfgs == generation cfgs for all variants")
        assert variants["no_ctx"].target_words == full["target_words"] == 180
        assert variants["no_ctx"].max_k == full["max_k"] == 15
        print(f"[selftest] {ds} variants single-factor; no_ctx holds target_words=180, max_k=15")

    # 3. functional window check on synthetic data (no SBERT: fixed embeddings)
    rng = np.random.default_rng(0)
    sents = [f"sentence number {i} discusses topic {i % 3} in some detail here" for i in range(30)]
    emb = rng.normal(size=(len(sents), 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cfg_full = ExtractConfig(**EXPECTED_FULL["arxiv"])
    from dataclasses import replace
    ranked = rank_sentences(sents, emb, cfg_full)
    ext_full = build_extract(sents, ranked, cfg_full)
    ext_noctx = build_extract(sents, ranked, replace(cfg_full, window=0))
    assert ext_noctx["expanded_ids"] == ext_noctx["core_ids"], "window=0 must not expand"
    assert set(ext_full["core_ids"]) < set(ext_full["expanded_ids"]), "window=1 must expand"
    print(f"[selftest] window functional check: core={len(ext_full['core_ids'])} sents, "
          f"window=1 expanded={len(ext_full['expanded_ids'])}, window=0 expanded={len(ext_noctx['expanded_ids'])}")

    print("[selftest] ALL PASSED — configs are clean single-factor. GPU generation may begin.")


if __name__ == "__main__":
    main()
