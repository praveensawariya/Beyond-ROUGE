# Human evaluation — rating sheets behind paper §8.2 (Table 6)

## Short answer

Yes, these ratings **are** used in the paper: they are the sole source of §8.2 and Table 6
(rater-averaged means, paired hybrid-vs-raw contrasts, and inter-rater agreement). No other rating
set is used for any system claim in the paper, and no rating from this set is used anywhere else.

## What is here

| File | Contents |
|---|---|
| `sheet_arxiv_rater1.csv`, `sheet_pubmed_rater1.csv` | Rater 1's returned sheets: 300 rows each (100 documents × 3 systems) |
| `sheet_arxiv_rater2.csv`, `sheet_pubmed_rater2.csv` | Rater 2's returned sheets: 300 rows each |
| `rater_instructions.md` | The instructions both raters received (identical file for both raters) |
| `deblinded.csv` | Every rating row with its system attached (1,200 rows), derived as described below |

Each sheet row is one summary: `idx` (document id), `label` (A/B/C), the summary text, and four
scores from 1 to 5: `fluency`, `coherence`, `informativeness`, `faithfulness`. The sheets are published
**exactly as returned** — no rating was edited, imputed, or dropped, including the fault described below.

## Blinding and deblinding (read this before reusing the labels)

Labels A/B/C are **randomised per document**: label A does not mean the same system in two documents.
The system for each row was recovered by exact string match of the summary against the study's system
predictions (each of the 600 summaries matched exactly one prediction: 100 documents × 3 systems per
dataset, systems `hybrid_ft`, `led_ftraw`, `extractive`). That per-row mapping is what `deblinded.csv`
records (`dataset, rater, idx, label, system, ...`); there is no global label-to-system code to apply.
Without `deblinded.csv`, the sheets alone cannot reproduce Table 6 — which is why it is included.

## How Table 6 is reproduced from `deblinded.csv`

Table 6 reports rater-averaged means (1–5) per dataset and system. Two details matter for an exact
reproduction:

1. **arXiv fluency is rater 1 only.** In the returned `sheet_arxiv_rater2.csv`, every row ends with a
   trailing comma, so the file has an extra empty column and the fluency column is blank for all 300 rows.
   The trailing values are preserved in `deblinded.csv`'s `extra_trailing` column but were **not** used:
   the paper ignores them rather than shifting them into fluency, and arXiv fluency in Table 6 rests on
   rater 1 alone. All other dimensions and all PubMed columns are complete.
2. **Rounding.** Five Table 6 means sit exactly on a `.xx5` boundary (arXiv led_ftraw faithfulness 2.945,
   arXiv extractive informativeness 2.925, PubMed hybrid_ft fluency 3.045, PubMed led_ftraw faithfulness
   2.855, PubMed extractive informativeness 2.945). The paper rounds these half-up (2.95, 2.93, 3.05, 2.86,
   2.95). Python's `round()` uses banker's rounding and gives values 0.01 lower on exactly those five cells;
   that is a rounding-convention difference, not a data difference. Every other Table 6 mean, and the paired
   hybrid-minus-raw faithfulness differences (arXiv +0.24, PubMed +0.07), reproduce exactly.

## Reliability — the paper's own verdict on these ratings

The paper reports this evaluation as **conducted but unreliable** and does **not** use it to support any
system claim (§8.2, §7): inter-rater Spearman is at chance on almost every dimension (0.00–0.08, except
PubMed fluency 0.26), exact agreement is 16–33%, rating marginals are near-uniform, and verbatim extractive
summaries average only about 3.0 on faithfulness. Every paired contrast in §8.2 includes zero. Anyone reusing
these sheets should carry those qualifications with them. Rater identity, the document-selection rule, and
any calibration round are not documented in the sheets or the instructions; raters are identified only as
`rater1` / `rater2` in the filenames, and no names or contact details appear anywhere in this folder.

## What is NOT here, and why

- **Source articles.** The 100-document article files supplied to raters (reference abstract + full article
  per document, ~10 MB, identical for both raters) are **not** included: they are the public `ccdv`
  arXiv/PubMed dataset texts, obtainable from the dataset itself, and they are not needed to reproduce any
  number in §8.2 or Table 6. The summary texts in the sheets are sufficient for the analysis.
- **The 40-document sheets from October 2026.** A separate, smaller rating set (40 documents per dataset)
  exists but was **not** pooled with this set and is not used in Table 6; it is not included here, to avoid
  mixing two evaluations.
- **System prediction files.** The predictions used for deblinding are the study's outputs, not code; they
  can be regenerated with `code/run_systems.py` (see the main README).
