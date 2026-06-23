#!/usr/bin/env python3
"""Combined Clinical + Molecular Model for MGUS Progression Prediction.

Builds a logistic regression combining the 200-probe SVM-RBF molecular risk
score with Mayo 20/2/20 clinical risk factors. Reports incremental AUC,
NRI, IDI, and decision curve analysis — mirroring the TissueCypher/Iyer 2022
and FR-PT Score validation frameworks.

NOTE: This script requires clinical variables that are NOT in the GEO deposit.
It will run once UAMS provides outcome labels + clinical data for GSE235356,
or once SWOG S0120 outcome labels are obtained for GSE122231.

Required clinical variables (Mayo 20/2/20):
  - m_protein_gL: serum M-protein level (g/L)
  - flc_ratio: serum free light chain ratio (kappa/lambda or lambda/kappa)
  - isotype: M-protein isotype (IgG vs non-IgG)
  - bm_plasma_pct: bone marrow plasma cell percentage (optional, for extended model)
  - progressed: binary outcome (0=stable, 1=progressed to MM)
  - time_to_event: time to progression or last follow-up (optional, for Cox extension)

Usage:
  python 06_combined_clinical_molecular.py --clinical-file <path_to_clinical_csv>

The clinical CSV must have a column matching GEO sample IDs to link with
molecular scores.
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Portable paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'analysis', 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE = 42


# ── Mayo 20/2/20 risk scoring ──────────────────────────────────────────────

def compute_mayo_score(df):
    """Compute Mayo 20/2/20 risk score from clinical variables.

    Mayo 2018 model (Rajkumar et al.):
      - M-protein >= 20 g/L (2.0 g/dL): 1 point
      - Free light chain ratio outside 0.26-1.65: 1 point (use > 20 as per 20/2/20)
      - Non-IgG isotype: 1 point
      - BM plasma cells >= 20%: 1 point (the "20" in 20/2/20 — optional)

    Total: 0-3 (or 0-4 with BM plasma %).
    """
    score = pd.Series(0, index=df.index, dtype=int)

    if 'm_protein_gL' in df.columns:
        score += (df['m_protein_gL'] >= 20).astype(int)

    if 'flc_ratio' in df.columns:
        # Abnormal if ratio > 20 (using the 20/2/20 threshold)
        # Some versions use outside 0.26-1.65; we use >20 per the 20/2/20 model
        score += (df['flc_ratio'] > 20).astype(int)

    if 'isotype' in df.columns:
        score += (df['isotype'] != 'IgG').astype(int)

    if 'bm_plasma_pct' in df.columns:
        score += (df['bm_plasma_pct'] >= 20).astype(int)

    return score


# ── Incremental value metrics ──────────────────────────────────────────────

def compute_nri(y_true, p_old, p_new, t=None):
    """Category-free Net Reclassification Improvement.

    NRI = P(p_new > p_old | event) + P(p_new < p_old | non-event)
        - P(p_new < p_old | event) - P(p_new > p_old | non-event)
    """
    events = y_true == 1
    non_events = y_true == 0

    # Events: correct reclassification is UP
    event_up = np.mean(p_new[events] > p_old[events])
    event_down = np.mean(p_new[events] < p_old[events])
    nri_events = event_up - event_down

    # Non-events: correct reclassification is DOWN
    nonevent_down = np.mean(p_new[non_events] < p_old[non_events])
    nonevent_up = np.mean(p_new[non_events] > p_old[non_events])
    nri_nonevents = nonevent_down - nonevent_up

    return nri_events + nri_nonevents


def compute_idi(y_true, p_old, p_new):
    """Integrated Discrimination Improvement.

    IDI = (mean(p_new|event) - mean(p_new|non-event))
        - (mean(p_old|event) - mean(p_old|non-event))
    """
    events = y_true == 1
    non_events = y_true == 0

    is_new = np.mean(p_new[events]) - np.mean(p_new[non_events])
    is_old = np.mean(p_old[events]) - np.mean(p_old[non_events])

    return is_new - is_old


def decision_curve_analysis(y_true, models_dict, thresholds=np.arange(0.01, 0.50, 0.01)):
    """Net benefit at each threshold probability.

    Net benefit = (TP/n) - (FP/n) * (pt / (1 - pt))
    where pt is the threshold probability.
    """
    n = len(y_true)
    results = {'threshold': thresholds}

    # Treat all
    prevalence = np.mean(y_true)
    treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    results['treat_all'] = treat_all
    results['treat_none'] = [0.0] * len(thresholds)

    for name, probs in models_dict.items():
        nb_list = []
        for pt in thresholds:
            pred_pos = probs >= pt
            tp = np.sum((pred_pos) & (y_true == 1))
            fp = np.sum((pred_pos) & (y_true == 0))
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
            nb_list.append(nb)
        results[name] = nb_list

    return pd.DataFrame(results)


# ── Bootstrap CI ───────────────────────────────────────────────────────────

def bootstrap_metric(y_true, metric_fn, n_boot=1000, seed=42):
    """Bootstrap 95% CI for any metric function that takes y_true and returns a scalar."""
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        vals.append(metric_fn(y_true, idx))
    return np.percentile(vals, [2.5, 97.5])


# ── Main analysis ──────────────────────────────────────────────────────────

def load_molecular_scores():
    """Load pre-computed molecular risk scores from SVM-RBF classifier.

    If model_performance.csv exists with per-sample predictions, use those.
    Otherwise, re-run the classifier to get probability scores.
    """
    # Check for existing per-sample scores
    scores_path = os.path.join(RESULTS_DIR, 'per_sample_molecular_scores.csv')
    if os.path.exists(scores_path):
        print(f"Loading molecular scores from {scores_path}")
        return pd.read_csv(scores_path, index_col=0)

    # If no pre-computed scores, we need to regenerate them
    print("No pre-computed molecular scores found.")
    print("Run 02_ml_classifier.py first, or provide per-sample scores.")
    print(f"Expected location: {scores_path}")
    print("\nThe file should have columns: sample_id, molecular_score, y_true, split (train/test)")
    sys.exit(1)


def main(clinical_file=None):
    print("=" * 70)
    print("Combined Clinical + Molecular Model Analysis")
    print("Mirroring TissueCypher (Iyer 2022) and FR-PT incremental value framework")
    print("=" * 70)

    # ── Load molecular scores ──
    mol = load_molecular_scores()
    print(f"\nMolecular scores loaded: {len(mol)} samples")

    if clinical_file is None:
        print("\n" + "!" * 70)
        print("NO CLINICAL DATA PROVIDED.")
        print("This script requires Mayo 20/2/20 clinical variables.")
        print("These are NOT in the GEO deposit for GSE235356.")
        print()
        print("To run this analysis, you need a CSV with columns:")
        print("  sample_id, m_protein_gL, flc_ratio, isotype, bm_plasma_pct,")
        print("  progressed, time_to_event (optional)")
        print()
        print("Sources for this data:")
        print("  1. UAMS team (request along with SWOG S0120 outcome labels)")
        print("  2. Bustoros/Ghobrial cohort (phs001323, via dbGaP)")
        print()
        print("Run with: python 06_combined_clinical_molecular.py --clinical-file <path>")
        print("!" * 70)

        # Generate a template CSV so the user knows the expected format
        template = pd.DataFrame({
            'sample_id': ['GSM7500908', 'GSM7500909', 'GSM7500910'],
            'm_protein_gL': [15.0, 25.0, 8.0],
            'flc_ratio': [1.2, 35.0, 0.8],
            'isotype': ['IgG', 'IgA', 'IgG'],
            'bm_plasma_pct': [5.0, 22.0, 3.0],
            'progressed': [0, 1, 0],
            'time_to_event': [1825, 730, 2190],
        })
        template_path = os.path.join(DATA_DIR, 'clinical_template.csv')
        template.to_csv(template_path, index=False)
        print(f"\nTemplate CSV written to: {template_path}")
        return

    # ── Load clinical data ──
    print(f"\nLoading clinical data from: {clinical_file}")
    clin = pd.read_csv(clinical_file, index_col='sample_id')

    # Merge molecular + clinical
    merged = mol.join(clin, how='inner')
    print(f"Merged: {len(merged)} samples with both molecular and clinical data")

    y = merged['progressed'].values
    mol_score = merged['molecular_score'].values

    # ── Model A: Clinical only (Mayo 20/2/20) ──
    mayo_score = compute_mayo_score(merged).values
    X_clinical = StandardScaler().fit_transform(mayo_score.reshape(-1, 1))

    lr_clinical = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
    # Use LOO-CV or 5-fold for small samples
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    p_clinical = np.zeros(len(y))
    for tr, te in cv.split(X_clinical, y):
        lr_clinical.fit(X_clinical[tr], y[tr])
        p_clinical[te] = lr_clinical.predict_proba(X_clinical[te])[:, 1]

    auc_clinical = roc_auc_score(y, p_clinical)
    print(f"\nModel A (Clinical only): AUC = {auc_clinical:.3f}")

    # ── Model B: Combined (Clinical + Molecular) ──
    X_combined = np.column_stack([X_clinical.ravel(), StandardScaler().fit_transform(
        mol_score.reshape(-1, 1)).ravel()])

    p_combined = np.zeros(len(y))
    for tr, te in cv.split(X_combined, y):
        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
        lr.fit(X_combined[tr], y[tr])
        p_combined[te] = lr.predict_proba(X_combined[te])[:, 1]

    auc_combined = roc_auc_score(y, p_combined)
    print(f"Model B (Clinical + Molecular): AUC = {auc_combined:.3f}")

    # ── Model C: Molecular only ──
    X_mol = StandardScaler().fit_transform(mol_score.reshape(-1, 1))
    p_mol = np.zeros(len(y))
    for tr, te in cv.split(X_mol, y):
        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
        lr.fit(X_mol[tr], y[tr])
        p_mol[te] = lr.predict_proba(X_mol[te])[:, 1]

    auc_mol = roc_auc_score(y, p_mol)
    print(f"Model C (Molecular only): AUC = {auc_mol:.3f}")

    # ── Incremental value: B vs A ──
    delta_auc = auc_combined - auc_clinical
    nri = compute_nri(y, p_clinical, p_combined)
    idi = compute_idi(y, p_clinical, p_combined)

    print(f"\n{'='*50}")
    print(f"INCREMENTAL VALUE (Model B vs Model A)")
    print(f"{'='*50}")
    print(f"  Delta AUC:  {delta_auc:+.3f} ({auc_clinical:.3f} -> {auc_combined:.3f})")
    print(f"  NRI:        {nri:.3f}")
    print(f"  IDI:        {idi:.3f}")

    # ── Bootstrap CIs ──
    print("\nBootstrapping 95% CIs (1000 iterations)...")
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y)
    boot_delta_auc, boot_nri, boot_idi = [], [], []

    for _ in range(1000):
        idx = rng.choice(n, n, replace=True)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_clin_b = roc_auc_score(y_b, p_clinical[idx])
            auc_comb_b = roc_auc_score(y_b, p_combined[idx])
            boot_delta_auc.append(auc_comb_b - auc_clin_b)
            boot_nri.append(compute_nri(y_b, p_clinical[idx], p_combined[idx]))
            boot_idi.append(compute_idi(y_b, p_clinical[idx], p_combined[idx]))
        except:
            continue

    ci_dauc = np.percentile(boot_delta_auc, [2.5, 97.5])
    ci_nri = np.percentile(boot_nri, [2.5, 97.5])
    ci_idi = np.percentile(boot_idi, [2.5, 97.5])

    print(f"  Delta AUC 95% CI: [{ci_dauc[0]:.3f}, {ci_dauc[1]:.3f}]")
    print(f"  NRI 95% CI:       [{ci_nri[0]:.3f}, {ci_nri[1]:.3f}]")
    print(f"  IDI 95% CI:       [{ci_idi[0]:.3f}, {ci_idi[1]:.3f}]")

    # ── Decision curve analysis ──
    dca = decision_curve_analysis(y, {
        'Clinical only (Mayo 20/2/20)': p_clinical,
        'Combined (Clinical + Molecular)': p_combined,
        'Molecular only': p_mol,
    })

    # ── Save results ──
    results = {
        'auc_clinical': float(auc_clinical),
        'auc_combined': float(auc_combined),
        'auc_molecular': float(auc_mol),
        'delta_auc': float(delta_auc),
        'delta_auc_ci': [float(ci_dauc[0]), float(ci_dauc[1])],
        'nri': float(nri),
        'nri_ci': [float(ci_nri[0]), float(ci_nri[1])],
        'idi': float(idi),
        'idi_ci': [float(ci_idi[0]), float(ci_idi[1])],
        'n_samples': int(n),
        'n_events': int(np.sum(y)),
        'prevalence': float(np.mean(y)),
    }

    results_path = os.path.join(RESULTS_DIR, 'combined_model_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    dca_path = os.path.join(RESULTS_DIR, 'decision_curve_analysis.csv')
    dca.to_csv(dca_path, index=False)
    print(f"Decision curve data saved to: {dca_path}")

    # ── Figures ──
    # ROC curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name, probs, color in [
        ('Clinical only (Mayo 20/2/20)', p_clinical, '#888888'),
        ('Combined (Clinical + Molecular)', p_combined, '#e74c3c'),
        ('Molecular only', p_mol, '#3498db'),
    ]:
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC: Clinical vs. Molecular vs. Combined', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_combined_roc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Decision curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(dca['threshold'], dca['treat_all'], 'k:', label='Treat all', alpha=0.5)
    ax.plot(dca['threshold'], dca['treat_none'], 'k--', label='Treat none', alpha=0.5)
    for name, color in [
        ('Clinical only (Mayo 20/2/20)', '#888888'),
        ('Combined (Clinical + Molecular)', '#e74c3c'),
        ('Molecular only', '#3498db'),
    ]:
        ax.plot(dca['threshold'], dca[name], label=name, color=color, linewidth=2)
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title('Decision Curve Analysis', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim([-0.05, max(dca['treat_all']) + 0.05])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_decision_curve.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nFigures saved to figures/")
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"\nKey result: Adding the 200-probe molecular score to Mayo 20/2/20")
    print(f"{'improved' if delta_auc > 0 else 'did not improve'} discrimination:")
    print(f"  AUC {auc_clinical:.3f} -> {auc_combined:.3f} (Delta = {delta_auc:+.3f})")
    print(f"  NRI = {nri:.3f}, IDI = {idi:.3f}")

    if delta_auc > 0.05:
        print("\n>>> SUBSTANTIAL incremental value. The molecular score adds")
        print("    meaningful prognostic information beyond clinical risk factors.")
        print("    This is the TissueCypher/Iyer 2022 result you need for payers.")
    elif delta_auc > 0:
        print("\n>>> MODEST incremental value. The molecular score adds some")
        print("    information but the improvement may not reach clinical significance.")
    else:
        print("\n>>> NO incremental value. The molecular score does not improve on")
        print("    clinical risk factors alone. Consider revising the model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combined Clinical + Molecular Model')
    parser.add_argument('--clinical-file', type=str, default=None,
                        help='Path to CSV with clinical variables (see template)')
    args = parser.parse_args()
    main(clinical_file=args.clinical_file)
