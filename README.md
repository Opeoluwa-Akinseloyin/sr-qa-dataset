# SR-QA

A question-answering dataset built from the **Characteristics of Included
Studies** tables of open-access systematic reviews.

Systematic reviewers read primary studies and record what they found in a table:
one row per included study, one column per thing they extracted — country,
design, sample size, intervention, outcomes. That table is an expert-curated
answer key. SR-QA turns each column into a question and each cell into its gold
answer, then pairs every item with the **full text of the source study the
reviewers read**.

So the task is: given a primary study, recover what the systematic reviewers
recorded about it.

**211,208 items · 3,203 reviews · 21,145 source papers**

## Splits

| file | items | reviews | source papers | use |
|---|--:|--:|--:|---|
| `data/train-2026.csv.gz` | 97,575 | 1,389 | 10,552 | training |
| `data/train-2025.csv.gz` | 51,123 | 780 | 5,530 | training |
| `data/dev-2026.csv.gz` | 14,616 | 247 | 1,640 | development |
| `data/test-2026.csv.gz` | 35,465 | 591 | 3,979 | **held-out test** |
| `data/test-pre-2025.csv.gz` | 12,429 | 196 | 1,376 | **held-out test** |

Splits are disjoint **by review**, so no review contributes to more than one
split and an answer key never leaks across the boundary.

### The two test sets are a contamination axis, not a duplicate

`test-2026` and `test-pre-2025` are split on the **review's** publication date —
before 2025 versus during 2026. The answer key lives in the review, so that is
the date that determines whether a model could have memorised the answers during
pretraining. The *source study* can be old in either arm; its year is irrelevant.

Report both. A model that scores well on `test-pre-2025` but poorly on
`test-2026` is likely recalling rather than reading.

## Quick start

```python
import pandas as pd

test = pd.read_csv("data/test-2026.csv.gz")
print(test[["question", "answer", "source_pmcid"]].head())

# all training data
train = pd.concat([pd.read_csv(f"data/train-{y}.csv.gz") for y in (2025, 2026)])
```

`manifest.json` carries per-split counts, byte sizes and SHA-256 checksums.

## Schema

One row per QA item.

| column | meaning |
|---|---|
| `review_pmcid` | the systematic review the answer key came from |
| `table_no` | 1-based index of the table within that review |
| `table_label`, `caption` | the table's own label and caption |
| `study_index`, `study_label` | which included study the row is about |
| `study_citations` | resolved ids (`PMCID:` / `PMID:` / `DOI:`) for that study |
| `header` | the table column the answer was read from |
| `question` | natural-language question for that column |
| `answer` | **gold** — verbatim cell text from the review's table |
| `source_pmcid` | the study to answer *from* — fetch this full text |
| `source_status` | how that link was resolved |
| `arm` | which split this row belongs to |

## Getting the source papers

Full texts are not redistributed here: mixed licences (see below) and ~2.8 GB.
Fetch by `source_pmcid` from the PMC Open Access subset:

```
https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC7005567
```

`review_context.jsonl.gz` gives each review's title, objectives and eligibility
criteria, which some setups supply as task context.

## Licence and attribution

The reviews this key derives from are **100% CC-BY** — verified from the
`<license>` element of all 5,434 review XML files; every one declares
`creativecommons.org/licenses/by`. **This dataset is redistributable with
attribution.**

Source study full texts carry mixed terms (`oa_commercial`, `oa_noncommercial`,
`oa_other`) and are *not* redistributable without a per-paper check, which is
why only their PMCIDs appear here.

## How the gold was built

Questions come from column headers; answers are copied verbatim from cells. No
model ever writes an answer.

An earlier version matched questions to columns **by the column's name**. Where a
table repeated a header — routine, since treatment and control columns are often
both labelled `Number of Subjects` — the first match won. So *"how many subjects
were in the control group?"* was answered with the treatment group's number.
Full-width section-heading rows were also read as studies.

The current version fixes this by showing an LLM the whole table (telling two
identical headers apart is impossible from names alone) and taking back only a
**structure map**: which rows are headers, which are not studies, which column
identifies the study, and one question per column *index*. Deterministic code
then copies each gold value from the cell the map points at, and this is checked
mechanically — every answer must be reconstructable from real cells of its own
column. **0 violations across 89,845 answers.**

Where the document's markup carries evidence it overrides the model: full-width
merged rows are banners, a column carrying `<xref ref-type="bibr">` links is the
study column, `<th>` rows are header rows.

## Known limitations

Please read these before treating this as a reference standard.

1. **The structure repair has not been human-audited.** Automatic checks confirm
   answers are verbatim and that the model did not contradict markup. They do
   **not** confirm it named columns correctly — a model that labels a treatment
   column "control" produces a map that passes every check. An audit is planned.
2. **12 tables still have two columns sharing one question.** Mostly the table
   itself provides nothing to separate them; one has fifteen consecutive columns
   flattening to the same header.
3. **~2.6% of items conflict by construction** — a study appearing in several row
   groups (arms, timepoints). The bulk are removed by an ambiguity filter; a
   residue remains.
4. **Gold is what the reviewers recorded, not ground truth.** Auditing found real
   errors in the published reviews — e.g. a total of `1048` where the source
   reports `1,068`. The dataset faithfully reproduces the review, mistakes
   included. Treat disagreement with a model as evidence about *either* side.
5. Answers are short verbatim strings, so exact match understates performance.
   A judged or fuzzy metric is recommended.

## Citation

```bibtex
@misc{srqa2026,
  title  = {SR-QA: Question Answering from Systematic Review Extraction Tables},
  author = {Akinseloyin, Ope},
  year   = {2026},
  note   = {Dataset. Derived from CC-BY systematic reviews in PubMed Central.}
}
```
