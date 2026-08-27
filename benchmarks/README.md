# Rocco Evaluation Benchmarks

This folder contains validation results from a peer-reviewed study comparing Rocco's automated evaluation accuracy against human expert evaluators.

## Overview

We evaluated Rocco's grading consistency by comparing its scores on five published DPM datasets against assessments from human domain experts. The study employed a cumulative link mixed model (CLMM) to statistically quantify differences.

## Contents

- **`data/`** — raw evaluation results and dataset citations; see `data/README.md` for the
  `evaluation_results.xlsx` sheet structure.

- **`figures/`** — data and analysis visualization
  - `1_proportion_of_scores_by_grader.png` — Aggregate distribution of rubric scores by all evaluators across all five evaluated datasets (Paper figure 1).
  - `2_proportion_of_scores_by_description.png` — Distribution of rubric item scores for each description and each evaluator.
  - `3_proportion_of_scores_by_rubric_item.png` — Aggregate score distributions by rubric items
  - `4_delta_clmm_by_rubric_item.png` — Leniency contrast by rubric item (Paper figure 2).

- **`statistical_evaluation.ipynb`** — Jupyter notebook with complete analysis code
  - Data loading and preprocessing
  - Krippendorff's alpha inter-rater reliability analysis
  - Bayesian Cumulative Link Mixed Model (CLMM) fitting and leniency contrast inference
  - Many Facet Item Response Theory (MFIRT) for faceted analysis


## Reproducibility

To re-run the analysis:

```bash
jupyter notebook statistical_evaluation.ipynb
```

## Citation

If you reference these evaluation benchmarks, cite both the Rocco software and the associated paper (DOI available in paper metadata).

---

*Study conducted: 2025-2026*
*Framework: Cumulative Link Mixed Model (CLMM) with Bayesian inference*
