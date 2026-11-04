# Beyond ROUGE: Efficiency, Content Quality, and Factual Faithfulness in Hybrid Scientific Document Summarization

Code for the paper **"Beyond ROUGE: An Empirical Study of Efficiency, Content Quality, and Factual Faithfulness
in Hybrid Scientific Document Summarization"** (Praveen Sawariya; revised manuscript, 2026-11-04).

This repository contains the study's pipeline code (version `v9.1-clean`), cleaned for public release:
Google Drive paths, Colab-only wording, and registry entries for a system excluded from the paper have been removed.
The analysis logic, defaults, and tuned configurations are unchanged. See `CLEANING.md` for exactly what was changed
and what was deliberately left out.

## What this code does

Eleven systems per dataset (arXiv and PubMed, 500 test documents each, seed 42) are scored on three axes:
content quality (ROUGE-1/2/L, BERTScore), factual faithfulness (Faith-ZS, a retrieval-filtered NLI scorer), and
computational efficiency (latency, peak GPU memory). The controlled core compares a fine-tuned LED-base model on
raw documents (`led_ftraw`) with the same model on SBERT+MMR extracts (`hybrid_ft`) — same backbone, data, and
budget, three seeds (7, 11, 23) — so the effect of extractive pre-filtering is isolated.

## Repository layout

| Path | Contents |
|---|---|
| `code/pipeline.py` | SBERT+MMR extraction, adaptive context expansion, LED generation (library module) |
| `code/data_utils.py` | Seeded dataset sampling from `ccdv/arxiv-summarization` / `ccdv/pubmed-summarization` (test split, seed 42, n = 500) |
| `code/train_ft.py` | Fine-tune LED-base: `--mode raw` (led_ftraw) or `--mode extracted` (hybrid_ft) |
| `code/run_systems.py` | Generate summaries for all systems on the fixed seeded sample (resumable) |
| `code/generate_from_checkpoint.py` | Generate from a fine-tuned checkpoint not in the `run_systems.py` registry |
| `code/analyze.py` | Score cached predictions: ROUGE, BERTScore, Faith-ZS, bootstrap CIs, Holm-corrected paired tests |
| `code/compare_paired.py` | Paired bootstrap comparison of two systems from `analyze.py` outputs |
| `code/faithfulness.py` | Faith-ZS scorer (3-sentence windows, stride 2, top-3 SBERT windows, DeBERTa-v3 NLI) |
| `code/faith_spec.py` | Write the formal Faith-ZS specification from `faithfulness.py`'s actual defaults |
| `code/bench.py`, `code/aggregate_efficiency.py` | Latency / peak-memory benchmarking (3 runs × 50 documents) and aggregation |
| `code/repair.py` | Verify-and-repair: drop sentences without NLI support or with numbers absent from the source (uses a *different* NLI model from Faith-ZS; see the paper §5.6) |
| `code/llm_baseline.py` | Zero-shot open-LLM baseline (Qwen2.5-7B-Instruct in the paper; needs a GPU with ≥ 24 GB) |
| `code/abstractiveness.py`, `code/learning_curve.py` | Abstractiveness statistics; training-scale (3k/6k/12k) sensitivity |
| `code/checkpoint_audit.py` | Checkpoint audit (CPU, downloads no model weights) — the audit behind the paper's baseline decisions |
| `code/gate3_selftest.py` | Self-test that ablation variants differ in exactly one configuration field |
| `code/repr_check.py` | Sample-representativeness check against the full test split |
| `code/make_figures.py` | Figures and trade-off tables from scored results (generates no model output) |
| `code/hyb_gpu.py` | GPU enforcement / environment check (fails loudly instead of silently running on CPU) |
| `configs/` | Tuned extractor configurations: arXiv λ = 0.75, lead bias 0.2; PubMed λ = 0.9, lead bias 0.06; target 180 words; plus single-factor ablation variants |
| `tests/test_smoke.py` | CPU-only smoke tests (no model downloads, no GPU) |
| `human_eval/` | Rating sheets behind paper §8.2 / Table 6 (100 documents per dataset, two raters), the rater instructions, and the deblinding mapping — see `human_eval/README.md` |

## Requirements

- Python 3.10+
- A GPU for generation, fine-tuning, scoring with Faith-ZS, and benchmarking. `run_systems.py` requires CUDA
  by default (`--device cuda`) and does not silently fall back to CPU.
- Install dependencies:

```bash
pip install -r requirements.txt
```

No credentials are needed for the public models and datasets used here. Do not put tokens, passwords, or private
paths in the code or in issues: model and dataset downloads use the public Hugging Face Hub defaults.

## Quickstart

```bash
# 0. CPU-only smoke test (no downloads, no GPU)
python -m unittest discover -s tests -v

# 1. Check the GPU environment
python code/hyb_gpu.py --check

# 2. Check that the evaluation sample is representative (CPU + dataset download)
python code/repr_check.py --dataset arxiv --n 500 --seed 42 \
  --cache_dir data_cache --out results/representativeness_arxiv.csv

# 3. Audit checkpoints before running baselines (CPU, no model weights downloaded)
python code/checkpoint_audit.py --out checkpoint_audit.json
```

## Controlled-core workflow (the paper's primary comparison)

```bash
# Fine-tune LED-base on raw documents and on extracts (same data, same budget).
# Paper settings: 3,000 pairs, 1 epoch, lr 3e-05, effective batch 16, seeds 7, 11, 23.
python code/train_ft.py --dataset arxiv --mode raw --n_train 3000 --epochs 1 \
  --lr 3e-05 --bs 1 --accum 16 --seed 7 --cache_dir data_cache --out models/led_ftraw_arxiv
python code/train_ft.py --dataset arxiv --mode extracted --n_train 3000 --epochs 1 \
  --lr 3e-05 --bs 2 --accum 8 --seed 7 --extract_cfg configs/best_extract_arxiv.json \
  --cache_dir data_cache --out models/led_ft_arxiv

# Generate summaries for the sweep on the fixed sample (follow the "Next:" hint it prints
# for scoring). Fine-tuned checkpoints live under models/; see code/run_systems.py.
python code/run_systems.py --dataset arxiv --n 500 --seed 42 \
  --systems lead extractive hybrid bart t5 led_only \
  --extract_cfg configs/best_extract_arxiv.json --out results/arxiv_n500

# Score, benchmark, and repair (examples; see each script's --help for all flags)
python code/analyze.py --results results/arxiv_n500 --systems lead extractive hybrid_ft led_ftraw
python code/bench.py --results results/arxiv_n500 --systems hybrid_ft led_ftraw --n 50 \
  --out results/bench_arxiv.csv
python code/repair.py --results results/arxiv_n500 --src_system hybrid_ft \
  --out_system hybrid_ft_repaired
python code/faith_spec.py --out faith_spec.txt
python code/make_figures.py --root results --out figures
```

Datasets are the public `ccdv` releases of Cohan et al. (2018). The 500-document sample uses seed 42 and is
identical across systems, so all primary tests are paired. Primary comparisons are pre-specified with Holm
correction; everything else in the analysis scripts is labelled descriptive or exploratory, as in the paper.

## What is NOT in this repository (read before citing the link)

- **Fine-tuned checkpoints.** Train them with `train_ft.py` (above), or place existing checkpoints under `models/`
  using the names in `code/run_systems.py` (`models/led_ft_arxiv/model`, and so on). Checkpoints are not committed
  (see `.gitignore`).
- **Prediction files, score tables, and the authors' results package.** These are the paper's reported outputs, not
  code; regenerate them with the workflow above.
- **Human-evaluation rating sheets.** These **are** included, in `human_eval/` (sheets, instructions, and
  deblinding) — see that folder's README for how Table 6 reproduces and for the paper's reliability verdict
  (conducted but unreliable; not used for any system claim). The 100-document source-article files are not
  included (public dataset texts; not needed for any §8.2 number).
- **The original Drive/Colab notebook.** The `.py` files here are the code that notebook embedded (version
  `v9.1-clean`), cleaned as described in `CLEANING.md`. The notebook itself, with its Drive mounts, is not included.
- **Dropped comparisons.** Training/generation scripts for a comparison dropped after the checkpoint audit are
  omitted; the audit script itself (`checkpoint_audit.py`) is included so the decision can be checked.

## Honest reproducibility note

This repository contains the study's code, cleaned for release. It does not by itself reproduce the paper's numbers
without GPU runs, fine-tuned checkpoints, and the authors' results package. Before citing this link in the paper,
run the smoke test and at least one end-to-end system on a small sample (for example `--n 20`) and compare the
output format against the paper's tables. If a small-sample run disagrees with the paper, that is a finding to report,
not something to adjust silently.

## Licence

MIT — see `LICENSE`. Copyright (c) 2026 Praveen Sawariya.
