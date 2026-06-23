# OSF Pre-Registration — Locked Classifier External Validation on SWOG S0120

**Project title:** External Validation of a Locked Gene Expression Classifier for Predicting Progression from MGUS and Smoldering Multiple Myeloma to Symptomatic Multiple Myeloma in the SWOG S0120 Cohort

**Principal Investigator:** Andrew Bouras, OMS-III, Nova Southeastern University Dr. Kiran C. Patel College of Osteopathic Medicine, ab4646@mynsu.nova.edu

**Faculty Advisor:** Robin J. Jacobs, Ph.D., M.S.W., M.S., M.P.H.

**External collaborator:** Brian Walker, Ph.D. (University of Miami Sylvester Comprehensive Cancer Center; SWOG Myeloma Translational Medicine Committee Chair)

**Date locked:** 2026-04-20

**Purpose of this registration:** to timestamp the classifier specification, feature set, decision threshold, and statistical analysis plan *before* outcome labels are received from SWOG. Any deviation from what is registered here must be disclosed in the final manuscript as a post-hoc analysis.

---

## 1. Classifier specification (locked)

- **Task:** binary classification of progression from MGUS or SMM to symptomatic multiple myeloma.
- **Input:** baseline CD138-selected bone marrow plasma cell gene expression, Affymetrix HG-U133 Plus 2.0 microarray, RMA-normalized.
- **Features:** 200 pre-selected probes (see `feature_importance.csv`). Feature set is frozen; no feature reselection on validation data.
- **Architecture:** soft-voting ensemble of a Support Vector Machine with radial basis function kernel (SVM-RBF) and a Logistic Regression (LR) model.
- **Hyperparameters (locked):** see `tuned_hyperparameters.json`.
  - SVM (RBF): C = 10.0, gamma = 0.001
  - Logistic Regression: C = 0.1
- **Decision threshold:** Youden-optimal threshold computed on the GSE235356 training cohort. Not re-optimized on SWOG S0120.
- **Training data:** GSE235356 (UAMS, 358 MGUS patients).
- **Held-out test performance (training publication):** AUROC 0.811 (95% CI 0.617 to 0.953), NPV 0.97 at the locked threshold.

---

## 2. Validation dataset

- **Source:** SWOG S0120 prospective observational trial of asymptomatic monoclonal gammopathies.
- **Expression data:** GEO accession GSE122231, n approximately 216 samples, Affymetrix HG-U133 Plus 2.0. Publicly available.
- **Outcome labels:** linked de-identified clinical outcomes to be released by SWOG/CRAB under an executed Data Use Agreement between SWOG and Nova Southeastern University. Outcome labels are **not yet in the investigators' possession** as of the lock date.
- **Probe overlap:** 200 of 200 classifier probes present on GPL570 platform (100%).

---

## 3. Primary hypothesis

The locked classifier, applied without modification to baseline GSE122231 expression data, will discriminate SWOG S0120 subjects who progressed to symptomatic multiple myeloma from those who did not, with an AUROC significantly greater than 0.50 (two-sided, alpha = 0.05).

---

## 4. Primary endpoint

AUROC for binary progression classification, with 1000-iteration stratified bootstrap 95% confidence interval.

## 5. Secondary endpoints

1. AUPRC with bootstrap 95% CI.
2. Sensitivity, specificity, PPV, NPV, accuracy at the locked threshold.
3. Calibration: intercept, slope, calibration plot, Brier score.
4. Decision curve analysis across clinically relevant threshold probabilities.
5. Subgroup performance by baseline diagnosis (MGUS vs SMM) and by Mayo 2018 (20/2/20) risk stratum.
6. Time-to-progression analyses: Kaplan-Meier curves by predicted risk tertile; Cox proportional hazards regression with classifier score as a continuous predictor and as risk tertiles.
7. Head-to-head AUROC comparison (DeLong test) against:
   - Mayo 2018 (20/2/20) clinical risk model
   - Mayo 2005 clinical risk model
   - GS36 gene signature (Sun et al., Haematologica 2023)
8. Combined clinical-plus-molecular model: incremental AUROC over Mayo 2018 alone.

---

## 6. Analysis plan (pre-specified)

1. Receive de-identified SWOG dataset under DUA.
2. QC the GSE122231 CEL files: RMA normalization, array quality metrics, exclusion of failing arrays per pre-specified criteria in script `04_external_validation_gse122231.py`.
3. Apply locked classifier; generate per-sample risk scores and binary predictions at the pre-locked threshold.
4. Compute primary and secondary endpoints as specified above.
5. Comparator analyses: apply Mayo 2005, Mayo 2018, and GS36 per `05_gs36_head_to_head.py` and `06_combined_clinical_molecular.py`.
6. Report per TRIPOD-AI (Collins et al., BMJ 2024). Include the completed TRIPOD-AI checklist as a supplementary file.

---

## 7. Software and environment

- Python 3.11: scikit-learn, numpy, pandas, lifelines, matplotlib.
- R 4.x: survival, rms (for calibration and decision curve analysis).
- All analyses to be run from version-controlled scripts in this pre-registration bundle. Any code change after lock must be disclosed.

---

## 8. Deviations policy

Any deviation from this pre-registration (feature change, threshold change, additional analysis, subgroup not listed above) will be disclosed in the manuscript as a post-hoc analysis and interpreted accordingly. Additions to secondary endpoints are permissible; primary hypothesis and primary endpoint cannot be altered post-hoc.

---

## 9. Cryptographic lock

The `LOCKED_MODEL_HASHES.txt` file in this bundle contains SHA-256 hashes of every file that defines the locked classifier and analysis pipeline:

| File | Role |
|---|---|
| `02_ml_classifier.py` | Model architecture and training code |
| `03_mps_scoring_system.py` | MPS (Module Scoring System) reference |
| `04_external_validation_gse122231.py` | External validation script (SWOG) |
| `05_gs36_head_to_head.py` | GS36 comparator pipeline |
| `06_combined_clinical_molecular.py` | Combined clinical+molecular model |
| `feature_importance.csv` | 200-probe feature list (locked) |
| `tuned_hyperparameters.json` | Tuned hyperparameters (locked) |

Repository git commit at lock time: **ba6eaef3fd7695243563ed29e1f3c1d6bba0a49a** (2026-04-04 04:33:07 -0400).

Any post-lock modification to these files produces a different SHA-256 and is automatically detectable.
