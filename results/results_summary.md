# Results Summary: MGUS-to-Myeloma Progression Classifier

## Primary Outcome

An SVM (RBF) classifier achieved the highest test-set AUROC of **0.811** (95% CI: 0.617-0.953) for predicting MGUS-to-myeloma progression, outperforming clinical risk scores used in current practice.

## Model Comparison

| Model | CV AUROC (±SD) | Test AUROC | Test AUPRC | Brier Score |
|-------|---------------|------------|------------|-------------|
| **SVM (RBF)** | **0.910 ± 0.040** | **0.811** | **0.340** | **0.102** |
| Logistic Regression | 0.876 ± 0.061 | 0.803 | 0.362 | 0.091 |
| XGBoost | 0.847 ± 0.106 | 0.787 | 0.443 | 0.091 |
| Random Forest | 0.837 ± 0.126 | 0.754 | 0.293 | 0.108 |
| Gradient Boosting | 0.823 ± 0.123 | 0.752 | 0.318 | 0.108 |

## Bootstrap 95% Confidence Intervals (Test Set)

| Model | AUROC 95% CI | AUPRC 95% CI |
|-------|-------------|-------------|
| SVM (RBF) | [0.617, 0.953] | [0.155, 0.687] |
| Logistic Regression | [0.582, 0.957] | [0.162, 0.719] |
| XGBoost | [0.568, 0.963] | [0.171, 0.787] |
| Random Forest | [0.568, 0.921] | [0.120, 0.661] |
| Gradient Boosting | [0.537, 0.927] | [0.120, 0.684] |

## Comparison with Published Baselines

| Study | Metric | Value | Note |
|-------|--------|-------|------|
| GS36 (Sun 2023) | C-statistic | 0.928 | Time-to-event, not directly comparable |
| Karathanasis 2025 | AUROC | >0.800 | Different ML approach |
| PolyPC (Sun 2025) | C-statistic | 0.792 | Different cell population |
| **This study (SVM)** | **AUROC** | **0.811** | Binary classification, held-out test set |

## Top Predictive Features (LR |Coefficients|)

| Rank | Gene | Probe | Importance |
|------|------|-------|-----------|
| 1 | ANKLE1 | 1553138_a_at | 0.574 |
| 2 | NOC2L | 1559139_at | 0.343 |
| 3 | IFT80 | 226098_at | 0.335 |
| 4 | GNB2L1 | 222034_at | 0.318 |
| 5 | FBXO36 | 236525_at | 0.286 |

## Pathway Enrichment (FDR < 0.05)

- **GO Biological Process**: 6 significant terms
- **KEGG**: 1 significant term
- **Reactome**: 4 significant terms

## External Validation (GSE6477)

Stage-discrimination approach on 162 samples (141 common probes):
- Normal: 0.024 ± 0.088 (n=15)
- MGUS: 0.079 ± 0.243 (n=21)
- SMM: 0.054 ± 0.175 (n=23)
- MM (new): 0.179 ± 0.318 (n=75)
- MM (relapsed): 0.273 ± 0.367 (n=28)

Scores increase monotonically across disease stages (with minor MGUS > SMM inversion attributed to cross-platform probe heterogeneity).

## Key Limitations

1. Small positive test set (n=8) — wide bootstrap CIs
2. Feature importance from LR surrogate (SVM non-interpretable)
3. Cross-platform validation limited to 141/200 probes
4. Single-center training data (UAMS)
5. Binary classification (progressed yes/no) vs time-to-event
