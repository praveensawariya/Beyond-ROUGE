# Cleaning log — from study code (v9.1-clean) to this public repository

## Source

The code here is the `code_v9` bundle (version `v9.1-clean`) embedded in the study notebook
`hybrid_v9_clean_free_tier__1.ipynb` (SETUP A cell), which wrote 23 Python files plus extractor/ablation
configuration JSONs. That notebook itself is **not** included: it mounts Google Drive and embeds the code as
base64, neither of which belongs in a public repository.

## Files kept (20 Python files, in `code/`)

`abstractiveness.py`, `aggregate_efficiency.py`, `analyze.py`, `bench.py`, `checkpoint_audit.py`,
`compare_paired.py`, `data_utils.py`, `faith_spec.py`, `faithfulness.py`, `gate3_selftest.py`,
`generate_from_checkpoint.py`, `hyb_gpu.py`, `learning_curve.py`, `llm_baseline.py`, `make_figures.py`,
`pipeline.py`, `repair.py`, `repr_check.py`, `run_systems.py`, `train_ft.py`,
plus `configs/best_extract_arxiv.json`, `configs/best_extract_pubmed.json`, and all six
`configs/variant_cfgs/*.json` files, copied unchanged.

## Files deliberately omitted (3)

- `train_longt5.py`, `generate_longt5.py` — scripts for a comparison that was **dropped** from the paper after
  the checkpoint audit found no documented arXiv/PubMed summarization checkpoint (paper §7). The audit script
  itself (`checkpoint_audit.py`) is kept, so the decision can be checked.
- `human_eval_analysis.py` — omitted because its docstring refers to a pre-registration file that is not in the
  code bundle and to a three-rater pack, neither of which matches the conducted evaluation reported in the paper
  (§8.2: 100 documents per dataset, two raters, reported as conducted but unreliable). Including it would
  misrepresent the study. No replacement human-evaluation script is provided; the conducted ratings are not used
  for any system claim in the paper.

## Changes made to the kept files (nothing else was changed)

1. **Paths.** All `/content/drive/MyDrive/hybrid_results/` prefixes were replaced with relative paths
   (`models/...`, `data_cache`, `results/...`). Fine-tuned checkpoint paths in `run_systems.py` are now
   `models/<name>/model`; place checkpoints there or train them with `train_ft.py`.
2. **Environment wording.** Docstrings and comments mentioning Google Drive or Colab were reworded to local
   equivalents ("locally", "a local directory", "an interrupted run", "a fresh session"). No logic was touched.
3. **Registry entries removed from `run_systems.py`.** Four entries (two model entries, two system entries) for a
   system excluded from the paper after a checkpoint audit were deleted, so the public registry contains only
   systems in the paper. No other model, system, default, or tuned value was changed.
4. **Stale help text fixed in `run_systems.py`.** `--extract_cfg` previously said "JSON from tune.py"; it now
   points to `configs/best_extract_*.json`. There is no `tune.py` in this repository; the tuned configurations
   are provided directly in `configs/`.
5. **Grammar.** Two comments produced by the wording replacements ("a interrupted run") were corrected to
   "an interrupted run".

In particular, the following were **not** changed: the extractor algorithm and defaults, the tuned configurations
(arXiv λ = 0.75, lead bias 0.2; PubMed λ = 0.9, lead bias 0.06; target 180 words — note the `ExtractConfig`
dataclass default of 220 is overridden by these tuned files), the Faith-ZS window/stride/top-k defaults
(3 / 2 / 3), the decoding settings (beam 4, 100–300 tokens, length penalty 2.0, no repeated trigrams),
the fine-tuning recipe, the scoring code, and all model identifiers for systems in the paper.

## Verification performed before release

- Every `.py` file compiles (`python -m py_compile`).
- `tests/test_smoke.py`: 6 CPU-only tests pass (text cleaning, sentence splitting, extraction with fake
  embeddings, tuned-configuration values, Faith-ZS sentence splitting, repair number regex). No model is
  downloaded and no GPU is needed for these tests.
- Searched the released `code/` directory for Drive paths, Colab imports, and credential-like strings
  (tokens, passwords, API keys): none found. (Occurrences of "tokenizer" / "tokens" are Hugging Face API names,
  not credentials.)
- The original notebook and the study manuscript were not modified by this cleaning; this repository is a copy
  with the changes listed above, not an edit of the originals.

## What still needs the author before the link goes in the paper

1. Upload this folder (not the zip) to GitHub and check that no checkpoint, prediction file, or rating sheet
   was added accidentally (`.gitignore` already excludes `models/`, `data_cache/`, `results/`, `*.jsonl`).
2. Run the smoke test and one small end-to-end system (`--n 20`) and confirm the output format matches the
   paper's tables.
3. Replace the placeholder licence holder/year in `LICENSE` if needed, and add the repository URL to the paper's
   Data and code availability section, replacing the current "no public repository URL is supplied" sentence.
