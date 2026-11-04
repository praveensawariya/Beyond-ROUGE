# Human Evaluation — Instructions for Raters

Thank you for helping with this study. You will rate summaries of scientific papers. You will **not** be told which version produced
which summary, and there are no right or wrong answers — your honest judgement
is the data.

## What you receive

- One rating sheet per dataset: `sheet_arxiv.csv` and/or `sheet_pubmed.csv`
  (open in Excel / Google Sheets / LibreOffice).
- The source articles for the sampled documents (ask the study owner for the
  article text for a given `idx` when you need it — you will need it mainly
  for the faithfulness rating).

Each row is one summary: `idx` (document id), `label` (A/B/C — a random label,
it does **not** identify the system), and the summary text. Fill in the four
score columns for every row. **Do not change `idx` or `label`.**

## The four dimensions (rate each 1–5)

**Fluency** — Is the text grammatically correct and easy to read?
5 = reads like a human wrote it · 3 = understandable but awkward/repetitive ·
1 = broken or garbled.

**Coherence** — Do the sentences connect logically and in a sensible order?
5 = flows like a well-structured abstract · 3 = ideas are present but jump
around · 1 = disconnected fragments.

**Informativeness** — Does the summary cover the paper's main problem, method,
and key results (compare against the source article / reference abstract)?
5 = all main points covered · 3 = some main points missing · 1 = misses the
point of the paper.

**Faithfulness** — Is **every** claim in the summary supported by the source
article? Watch especially for invented numbers, results, methods, or claims
that contradict the source.
5 = everything is supported · 3 = minor unsupported details · 1 = contains
invented or contradicted facts.

## Rules

- Rate every row; leave no score blank. If a summary is empty or nonsensical,
  score it 1 on all dimensions.
- Rate independently — do not discuss scores with the other rater while rating.
- Judge the summary against the **source article**, not against what you think
  a good summary should ideally contain beyond the source.
- Take breaks. 120 summaries is a lot; accuracy drops when rushed. Roughly
  2–3 minutes per summary is normal.

## When you finish

Save the sheet as CSV with your name in the filename, e.g.
`sheet_arxiv_rater1.csv`, and return it to the study owner. Do not edit
anything except the four score columns.
