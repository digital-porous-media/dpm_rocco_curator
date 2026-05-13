# Rocco Evaluation Benchmarks

This folder contains validation results from a peer-reviewed study comparing Rocco's automated evaluation accuracy against human expert evaluators.

## Overview

We evaluated Rocco's grading consistency by comparing its scores on five published DPM datasets against assessments from human domain experts. The study employed a cumulative link mixed model (CLMM) to statistically quantify differences.

## Contents

- **`data/`** — Raw evaluation data and statistics
  - `evaluation_results.xlsx` — 300 scores across 6 sheets:
    - 5 per-description sheets (Description1–Description5): Rows = 10 rubric items, Columns = 6 evaluators (Rocco + 5 humans)
    - 1 aggregated flat-file sheet: Flat format with columns for Description, Rubric Item, Grader, Score (300 rows total)
  - `dataset_descriptions.md` — Full citations and original descriptions for the 5 evaluated datasets.

- **`figures/`** — Publication-ready figures
  - `1_proportion_of_scores_by_grader.png` — Aggregate distribution of rubric scores by all evaluators across all five evaluated datasets (Paper figure 1).
  - `2_proportion_of_scores_by_description.png` — Distribution of rubric item scores for each description and each evaluator.
  - `3_proportion_of_scores_by_rubric_item.png` — Aggregate score distributions by rubric items
  - `4_delta_clmm_by_rubric_item.png` — Leniency contrast by rubric item (Paper figure 2).

- **`statistical_evaluation.ipynb`** — Jupyter notebook with complete analysis code
  - Data loading and preprocessing
  - Krippendorff's alpha inter-rater reliability analysis
  - Bayesian Cumulative Link Mixed Model (CLMM) fitting and leniency contrast inference
  - Many Facet Item Response Theory (MFIRT) for faceted analysis
  - Figure generation and statistical validation
  - CLMM and MFIRT results (leniency contrasts, credible intervals) computed and displayed inline

## Key Findings

**Rocco's Median Leniency Contrast:** -0.024 (95% CI: [-0.493, 0.306])

This indicates Rocco is marginally stricter than humans, but the credible interval centered near zero shows no meaningful systematic bias.

**Per-Item Analysis:** Rocco was slightly stricter on rubric items 2, 4, 5, and 8, but divergences are small and explained by rubric ambiguity or literal application differences rather than fundamental disagreement.

## Reproducibility

To re-run the analysis:

```bash
jupyter notebook analysis.ipynb
```

The notebook is self-contained with all data loading, preprocessing, and visualization steps clearly documented.

## Citation

If you reference these evaluation benchmarks, cite both the Rocco software and the associated peer-reviewed paper (DOI available in paper metadata).

---

*Study conducted: 2025-2026*
*Framework: Cumulative Link Mixed Model (CLMM) with Bayesian inference*
