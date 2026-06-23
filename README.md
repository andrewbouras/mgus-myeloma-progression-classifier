# Locked Gene-Expression Classifier for MGUS/SMM → Multiple Myeloma Progression

Pre-registered, pre-locked external-validation pipeline for a 200-probe machine-learning
classifier predicting progression from MGUS / smoldering multiple myeloma to symptomatic
multiple myeloma. This repository exists to make the **pre-locking verifiable**: the model
specification, hyperparameters, decision threshold, and statistical analysis plan were
frozen *before* any external outcome data were received.

> Status: external validation in the SWOG S0120 cohort (GEO: GSE122231) is pending an
> executed Data Use Agreement with SWOG/CRAB. This repository is the public, IP-safe
> portion of the pre-registration bundle.

## What proves the analysis was pre-locked

- **`prereg/LOCKED_MODEL_HASHES.txt`** — SHA-256 hashes of every file that defines the
  locked classifier and pipeline (model code, scoring system, validation script, GS36
  comparator, combined model, the 200-probe feature list, and hyperparameters). Anyone can
  recompute these hashes against the released files to confirm nothing changed after the
  lock. Files held back for IP reasons (below) are still covered by these hashes, so they
  remain verifiable when released.
- **`prereg/OSF_PREREGISTRATION.md`** — the locked classifier specification, primary/secondary
  endpoints, and the pre-specified analysis plan.
- **Git history of this repository** — the initial commit timestamps this public bundle.

Verify the lock:

```bash
cd prereg && sha256sum -c LOCKED_MODEL_HASHES.txt   # for files present locally
```

## Repository layout

```
prereg/
  OSF_PREREGISTRATION.md     # locked spec + analysis plan
  LOCKED_MODEL_HASHES.txt    # SHA-256 lock over the full bundle (incl. embargoed files)
  tuned_hyperparameters.json # locked hyperparameters
scripts/
  01_download_geo_data.py            # GEO retrieval
  02_ml_classifier.py                # model architecture / training (loads feature list at runtime)
  04_external_validation_gse122231.py# external-validation execution script (SWOG S0120)
  05_gs36_head_to_head.py            # GS36 (Sun 2023) comparator
  06_combined_clinical_molecular.py  # combined clinical + molecular model
results/
  results_summary.md         # aggregate, de-identified performance metrics
requirements.txt
```

## What is intentionally NOT in this public repository (embargoed)

To protect the commercializable core (NanoString-panel translation / LDT pathway), the
following are **held in a private release shared with SWOG under the DUA**, not published here.
They remain covered by `LOCKED_MODEL_HASHES.txt`, so their integrity is independently
verifiable on release:

- `feature_importance.csv` — the locked **200-probe feature list**.
- `03_mps_scoring_system.py` — the **MPS module composition** (gene→probe mapping).
- Any trained model weights, patient-level expression/phenotype/score tables, and all
  SWOG-derived outcome data.

No protected health information is contained in this repository. The associated study was
determined Not Human Subjects Research (NSU IRB 2026-239-NSU).

## Datasets

- **Training:** GSE235356 (UAMS, 358 MGUS patients), Affymetrix HG-U133 Plus 2.0.
- **External validation:** GSE122231 (SWOG S0120, ~216 CD138-selected samples), same platform;
  linked longitudinal progression outcomes provided by SWOG under DUA.

## Reporting

Analyses are reported per TRIPOD-AI (Collins et al., BMJ 2024).

## Citation / contact

Andrew Bouras, OMS-III — Nova Southeastern University, Dr. Kiran C. Patel College of
Osteopathic Medicine. SWOG investigators and statisticians are collaborators and co-authors
on resulting work.
