# Round-2 revision analyses (November 2026)

New analyses added for the second round of peer review of *Beyond ROUGE*.
They are reported in the manuscript's §5.2 (document-level correlation) and
§5.5c (copy-adjusted Faith-ZS), and in Figures 1–2. No new model runs were
performed for them: they use per-sample tables and predictions already on
file from the main study.

## Contents

| File | What it is |
|---|---|
| `analysis_doclevel_corr.py` | Document-level (n = 500/system) Spearman correlation between ROUGE-1 and Faith-ZS on arXiv, with bootstrap 95% CIs; also recomputes Kendall's τ = −0.21 over the 8 Table-1 systems. |
| `doclevel_corr_arxiv.csv` | Its output (11 systems + pooled row). |
| `analysis_copy_adjusted.py` | Copy-adjusted Faith-ZS for the arXiv fine-tuned pair (hybrid_ft − led_ftraw): raw, length-adjusted, abstractiveness-adjusted, both-adjusted, and stratified by the abstractiveness gap. |
| `copy_adjusted_arxiv.csv` | Its output (per-document differences). |
| `abstractiveness_perdoc_arxiv_hybrid_ft.csv`, `abstractiveness_perdoc_arxiv_led_ftraw.csv` | Per-document novel-bigram / copy-sentence rates recomputed with the repo's `code/abstractiveness.py` (`doc_metrics`) on the stored arXiv predictions. |
| `make_figs.py` | Regenerates Figures 1–2 (ROUGE-1 vs Faith-ZS) with plain system names, a regime legend, and manual label offsets. |
| `fig1_arxiv_revised.png`, `fig2_pubmed_revised.png` | The regenerated figures. |

## Key results

- Document-level ROUGE-1 vs Faith-ZS Spearman, arXiv: −0.06 to +0.24 per
  system (nine of eleven 95% CIs include zero); pooled over 5,500
  document–system pairs: −0.173 [−0.198, −0.148].
- Fine-tuned-pair Faith-ZS difference (hybrid_ft − led_ftraw, ×100, n = 500):
  raw +0.50 [−1.85, +2.87]; abstractiveness-adjusted +6.10 [+3.54, +8.80];
  slope −1.08 Faith-ZS points per novel-bigram point. Stratified by the
  abstractiveness gap: +11.72 / −0.58 / −9.65.
- Limits (stated in §5.5c): the adjustment is linear and the intercept is
  interpolation; the per-document novel-bigram means sit ~1.5 points above
  the manuscript's §5.5 values because the test-cache article copy differs
  from the main-study copy (the paired difference reproduces Table A1:
  +5.18 vs +5.16); a residual Faith-ZS difference is not a measurement of
  truth. The PubMed version of the copy-adjusted analysis could not be run:
  no PubMed per-sample table or PubMed source-article copy is available.

## Inputs (not in this repo)

The scripts read the authors' materials, which are not committed here
(checkpoints, prediction files, and per-sample score tables are excluded
by design — see the root README):

- `arxiv_n500_per_sample.csv` — per-sample ROUGE/BERTScore/Faith-ZS table
  (500 arXiv docs × 11 systems; the excluded system is dropped by the
  script). Needed by both analysis scripts.
- `table_main_arxiv.csv`, `table_main_pubmed.csv` — the cached full
  system tables with 95% CIs. Needed by `make_figs.py`.

Point `BEYOND_ROUGE_DATA` at the folder holding them, e.g.:

```bash
export BEYOND_ROUGE_DATA=/path/to/authors/materials
python3 analysis_doclevel_corr.py        # writes doclevel_corr_arxiv.csv
python3 analysis_copy_adjusted.py        # writes copy_adjusted_arxiv.csv
python3 make_figs.py                     # writes fig1_arxiv_revised.png, fig2_pubmed_revised.png
```

The committed CSVs and PNGs are the outputs of exactly these scripts run
on those inputs; re-running them reproduces the committed files.
