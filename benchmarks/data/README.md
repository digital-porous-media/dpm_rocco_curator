# Benchmark Data

Raw data files from the Rocco evaluation benchmarking study.

## Files

### `evaluation_results.xlsx`
Excel workbook containing 300 individual scores (6 evaluators × 5 descriptions × 10 rubric items):

**Sheet structure:**
- **5 description sheets** (one per evaluated description):
  - `Description1` (Armstrong et al., 2025) — Multimineral Model
  - `Description2` (Vidal et al., 2024) — Brazilian pre-salt carbonates
  - `Description3` (Wang et al., 2023) — Brine-nitrogen co-injection
  - `Description4` (Chen et al., 2019) — *[See dataset_descriptions.md]*
  - `Description5` (Guiltinan et al., 2020) — *[See dataset_descriptions.md]*

  Each sheet contains:
  - Rows: 10 rubric items
  - Columns: Evaluator scores (Human 1–5, Rocco)
  - Values: Scores on ordinal 0–1 scale (0 = poor, 0.5 = adequate, 1 = good)

- **1 aggregated flat-file sheet** (`flat_file`):
  - Flat file format with columns: Description, Rubric Item, Grader, Score
  - 300 rows (6 evaluators × 5 descriptions × 10 items)
  - Optimized for statistical modeling (Krippendorff's alpha, CLMM, MFIRT)


### `dataset_descriptions.md`
Full citations for the 5 evaluated datasets with DOI/URL references.

---

**For questions or corrections**, refer to the Jupyter notebook in `../statistical_evaluation.ipynb` for complete data processing, inter-rater reliability, CLMM, and MFIRT analysis code.
