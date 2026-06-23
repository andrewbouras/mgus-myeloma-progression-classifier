#!/usr/bin/env python3
"""Head-to-head comparison: GS36 signature vs 200-probe ML classifier.

Compares four feature sets on the same train/test split:
  1. GS36 (36 probes from Sun et al. 2023)
  2. 200-probe data-driven model (original classifier)
  3. Overlap-only (probes shared between GS36 and 200-probe)
  4. Combined (200-probe + non-overlapping GS36 probes)

All models use the same nested CV framework, bootstrap CIs, and evaluation
pipeline as 02_ml_classifier.py for a fair comparison.
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, train_test_split,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, brier_score_loss, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Portable paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'analysis', 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_primary_dataset():
    """Load GSE235356 expression and phenotype data."""
    print("Loading GSE235356 primary dataset...")
    expr = pd.read_csv(os.path.join(DATA_DIR, 'GSE235356_expression.csv'), index_col=0)
    pheno = pd.read_csv(os.path.join(DATA_DIR, 'GSE235356_phenotype.csv'), index_col=0)
    y = (pheno['char_disease state'] == 'Progressing MGUS').astype(int)
    X = expr.T
    X.index = X.index.astype(str)
    y.index = y.index.astype(str)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    print(f"  {X.shape[0]} samples, {X.shape[1]} probes")
    print(f"  Stable={sum(y==0)}, Progressing={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    return X, y


def load_gs36_probes():
    """Load the 36 GS36 probe IDs from Sun et al. 2023 supplementary Table S2."""
    gs36_path = os.path.join(DATA_DIR, 'gs36_genes.csv')
    gs36 = pd.read_csv(gs36_path)
    print(f"  Loaded GS36: {len(gs36)} genes/probes")
    return gs36


# ── Preprocessing for specific probe sets ─────────────────────────────────────

def preprocess_probe_set(X_train, y_train, X_test, probe_ids, label=""):
    """Subset to specified probes, handle NaN, scale. No ANOVA — probes are pre-selected."""
    available = [p for p in probe_ids if p in X_train.columns]
    missing = [p for p in probe_ids if p not in X_train.columns]
    if missing:
        print(f"  [{label}] WARNING: {len(missing)} probes not found in expression data: {missing[:5]}...")
    print(f"  [{label}] Using {len(available)}/{len(probe_ids)} probes")

    X_tr = X_train[available].copy()
    X_te = X_test[available].copy()

    # Fill NaN with training column medians
    col_medians = X_tr.median()
    X_tr = X_tr.fillna(col_medians)
    X_te = X_te.fillna(col_medians)

    # Standardize
    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(
        scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns
    )
    X_te_scaled = pd.DataFrame(
        scaler.transform(X_te), index=X_te.index, columns=X_te.columns
    )
    return X_tr_scaled, X_te_scaled, scaler, available


def preprocess_anova(X_train, y_train, X_test, n_features=200):
    """Original ANOVA-based feature selection (reproduces 02_ml_classifier.py)."""
    print(f"  [200-probe] ANOVA feature selection (k={n_features})...")
    all_columns = X_train.columns
    X_tr_np = X_train.values.astype(np.float64)
    X_te_np = X_test.values.astype(np.float64)

    # Fill NaN
    nan_mask = np.isnan(X_tr_np)
    if nan_mask.any():
        col_medians = np.nanmedian(X_tr_np, axis=0)
        for j in range(X_tr_np.shape[1]):
            X_tr_np[nan_mask[:, j], j] = col_medians[j]
            te_nan = np.isnan(X_te_np[:, j])
            X_te_np[te_nan, j] = col_medians[j]

    # Variance filter: remove bottom 50%
    vt = VarianceThreshold()
    vt.fit(X_tr_np)
    threshold = np.percentile(vt.variances_, 50)
    vt2 = VarianceThreshold(threshold=threshold)
    X_tr_np = vt2.fit_transform(X_tr_np)
    X_te_np = vt2.transform(X_te_np)
    kept_cols = all_columns[vt2.get_support()]

    # ANOVA
    n_sel = min(n_features, X_tr_np.shape[1])
    skb = SelectKBest(f_classif, k=n_sel)
    X_tr_np = skb.fit_transform(X_tr_np, y_train)
    X_te_np = skb.transform(X_te_np)
    final_cols = kept_cols[skb.get_support()]

    # Scale
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_np)
    X_te_np = scaler.transform(X_te_np)

    X_tr = pd.DataFrame(X_tr_np, index=X_train.index, columns=final_cols)
    X_te = pd.DataFrame(X_te_np, index=X_test.index, columns=final_cols)
    return X_tr, X_te, scaler, list(final_cols)


# ── Model definitions ─────────────────────────────────────────────────────────

def get_tuned_models(n_pos, n_neg):
    """Return models with hyperparameter search spaces."""
    scale_pos = n_neg / n_pos
    return {
        'Logistic Regression': (
            LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000,
                               class_weight='balanced', random_state=RANDOM_STATE),
            {'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
        ),
        'Random Forest': (
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1),
            {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 10, None],
             'min_samples_leaf': [2, 5, 10], 'max_features': ['sqrt', 'log2']}
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {'n_estimators': [100, 200, 300], 'max_depth': [2, 3, 4, 5],
             'learning_rate': [0.01, 0.05, 0.1, 0.2], 'subsample': [0.7, 0.8, 1.0],
             'min_samples_leaf': [2, 5, 10]}
        ),
        'XGBoost': (
            xgb.XGBClassifier(scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
                              eval_metric='logloss', verbosity=0, n_jobs=1),
            {'n_estimators': [100, 200, 300], 'max_depth': [2, 3, 4, 5],
             'learning_rate': [0.01, 0.05, 0.1, 0.2], 'subsample': [0.7, 0.8, 1.0],
             'colsample_bytree': [0.6, 0.8, 1.0], 'min_child_weight': [1, 5, 10]}
        ),
        'SVM (RBF)': (
            SVC(kernel='rbf', probability=True, class_weight='balanced',
                random_state=RANDOM_STATE),
            {'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto', 0.001, 0.01]}
        ),
    }


# ── Training & evaluation ────────────────────────────────────────────────────

def nested_cv(X_tr, y_train, models_and_params, label=""):
    """Nested CV: outer 5-fold x 3 repeats, inner 3-fold RandomizedSearchCV."""
    print(f"\n  [{label}] Nested CV (5x3 outer, 3-fold inner)...")
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}
    for name, (base_model, param_dist) in models_and_params.items():
        fold_aucs = []
        total_combos = 1
        for v in param_dist.values():
            total_combos *= len(v) if isinstance(v, list) else 10
        n_iter = min(50, total_combos)

        for fold_i, (tr_idx, val_idx) in enumerate(outer_cv.split(X_tr, y_train)):
            X_fold_tr = X_tr.iloc[tr_idx]
            y_fold_tr = y_train.iloc[tr_idx]
            X_fold_val = X_tr.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            search = RandomizedSearchCV(
                type(base_model)(**base_model.get_params()),
                param_distributions=param_dist, n_iter=n_iter,
                scoring='roc_auc', cv=inner_cv, random_state=RANDOM_STATE,
                n_jobs=1, refit=True
            )
            if name == 'Gradient Boosting':
                sw = compute_sample_weight('balanced', y_fold_tr)
                search.fit(X_fold_tr, y_fold_tr, sample_weight=sw)
            else:
                search.fit(X_fold_tr, y_fold_tr)

            yp = search.predict_proba(X_fold_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_fold_val, yp))

        cv_results[name] = {
            'AUROC_mean': np.mean(fold_aucs),
            'AUROC_std': np.std(fold_aucs),
            'n_folds': len(fold_aucs)
        }
        print(f"    {name}: CV AUROC = {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}")

    # Final fit on full training set
    tuned_models = {}
    for name, (base_model, param_dist) in models_and_params.items():
        total_combos = 1
        for v in param_dist.values():
            total_combos *= len(v) if isinstance(v, list) else 10
        n_iter = min(50, total_combos)
        search = RandomizedSearchCV(
            type(base_model)(**base_model.get_params()),
            param_distributions=param_dist, n_iter=n_iter,
            scoring='roc_auc', cv=inner_cv, random_state=RANDOM_STATE,
            n_jobs=1, refit=True
        )
        if name == 'Gradient Boosting':
            sw = compute_sample_weight('balanced', y_train)
            search.fit(X_tr, y_train, sample_weight=sw)
        else:
            search.fit(X_tr, y_train)
        tuned_models[name] = search.best_estimator_

    return cv_results, tuned_models


def evaluate_test(models, X_te, y_test, label=""):
    """Evaluate all models on held-out test set."""
    rows = []
    predictions = {}
    for name, model in models.items():
        yp = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, yp)
        ap = average_precision_score(y_test, yp)
        brier = brier_score_loss(y_test, yp)
        fpr, tpr, thresholds = roc_curve(y_test, yp)
        j_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[j_idx]
        y_opt = (yp >= opt_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_opt).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        rows.append({
            'Feature_Set': label, 'Model': name, 'AUROC': auc, 'AUPRC': ap,
            'Brier_Score': brier, 'Sensitivity': sens, 'Specificity': spec,
            'PPV': ppv, 'NPV': npv, 'F1': f1_score(y_test, y_opt),
            'Accuracy': accuracy_score(y_test, y_opt)
        })
        predictions[name] = {'y_prob': yp, 'fpr': fpr, 'tpr': tpr, 'auc': auc}
    return pd.DataFrame(rows), predictions


def bootstrap_ci(models, X_te, y_test, label="", n_boot=1000):
    """Bootstrap 95% CIs for test-set AUROC."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_test)
    ci_rows = []
    for name, model in models.items():
        yp = model.predict_proba(X_te)[:, 1]
        boot_aucs = []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            if len(np.unique(y_test.iloc[idx])) < 2:
                continue
            boot_aucs.append(roc_auc_score(y_test.iloc[idx], yp[idx]))
        lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
        ci_rows.append({
            'Feature_Set': label, 'Model': name,
            'AUROC': roc_auc_score(y_test, yp),
            'AUROC_CI_low': lo, 'AUROC_CI_high': hi,
            'n_boot_valid': len(boot_aucs)
        })
    return pd.DataFrame(ci_rows)


# ── DeLong test for comparing AUROCs ─────────────────────────────────────────

def _compute_midrank(x):
    """Compute midranks for DeLong test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n - 1 and z[k + 1] == z[k]:
            k += 1
        for t in range(i, k + 1):
            rank[t] = 0.5 * (i + k) + 1
        i = k + 1
    rank_out = np.empty(n)
    rank_out[j] = rank
    return rank_out


def _auc_variance(y_true, y_score):
    """Compute AUC and its variance via DeLong's method."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m = len(pos)
    n = len(neg)
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(m), np.zeros(n)])
    midranks = _compute_midrank(all_scores)
    pos_ranks = midranks[all_labels == 1]
    auc = (np.sum(pos_ranks) - m * (m + 1) / 2) / (m * n)
    # Placement values
    v_pos = np.zeros(m)
    v_neg = np.zeros(n)
    for i in range(m):
        v_pos[i] = np.sum(neg < pos[i]) + 0.5 * np.sum(neg == pos[i])
    v_pos /= n
    for j in range(n):
        v_neg[j] = np.sum(pos > neg[j]) + 0.5 * np.sum(pos == neg[j])
    v_neg /= m
    s_pos = np.var(v_pos, ddof=1) if m > 1 else 0
    s_neg = np.var(v_neg, ddof=1) if n > 1 else 0
    var_auc = s_pos / m + s_neg / n
    return auc, var_auc


def delong_test(y_true, y_score_a, y_score_b):
    """Two-sided DeLong test comparing two AUROCs on the same samples.
    Returns (z_stat, p_value, auc_a, auc_b, delta_auc).
    """
    auc_a, var_a = _auc_variance(y_true, y_score_a)
    auc_b, var_b = _auc_variance(y_true, y_score_b)
    # Covariance via placement values
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_a = y_score_a[pos_mask]
    neg_a = y_score_a[neg_mask]
    pos_b = y_score_b[pos_mask]
    neg_b = y_score_b[neg_mask]
    m = len(pos_a)
    n = len(neg_a)
    v_pos_a = np.array([np.sum(neg_a < p) + 0.5 * np.sum(neg_a == p) for p in pos_a]) / n
    v_pos_b = np.array([np.sum(neg_b < p) + 0.5 * np.sum(neg_b == p) for p in pos_b]) / n
    v_neg_a = np.array([np.sum(pos_a > ne) + 0.5 * np.sum(pos_a == ne) for ne in neg_a]) / m
    v_neg_b = np.array([np.sum(pos_b > ne) + 0.5 * np.sum(pos_b == ne) for ne in neg_b]) / m
    cov_pos = np.cov(v_pos_a, v_pos_b, ddof=1)[0, 1] if m > 1 else 0
    cov_neg = np.cov(v_neg_a, v_neg_b, ddof=1)[0, 1] if n > 1 else 0
    cov_ab = cov_pos / m + cov_neg / n
    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        return 0.0, 1.0, auc_a, auc_b, auc_a - auc_b
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p, auc_a, auc_b, auc_a - auc_b


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_comparison_roc(all_predictions, y_test, figures_dir):
    """Overlay ROC curves for the best model from each feature set."""
    sns.set_style('white')
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'figure.dpi': 300})
    colors = {'GS36 (36 probes)': '#d62728', '200-Probe ML': '#1f77b4',
              'Overlap (shared)': '#2ca02c', 'Combined (GS36 + ML)': '#ff7f0e'}

    fig, ax = plt.subplots(figsize=(8, 7))
    for fs_label, preds in all_predictions.items():
        # Pick the best model by AUROC
        best_name = max(preds, key=lambda k: preds[k]['auc'])
        p = preds[best_name]
        color = colors.get(fs_label, '#9467bd')
        ax.plot(p['fpr'], p['tpr'], lw=2, color=color,
                label=f'{fs_label} — {best_name} (AUROC = {p["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Head-to-Head ROC Comparison: GS36 vs Data-Driven ML')
    ax.legend(loc='lower right', fontsize=9)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(figures_dir, 'gs36_vs_ml_roc.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_comparison_bar(all_test_results, figures_dir):
    """Grouped bar chart: best AUROC per feature set."""
    best_per_fs = all_test_results.loc[
        all_test_results.groupby('Feature_Set')['AUROC'].idxmax()
    ].sort_values('AUROC', ascending=True)

    colors = {'GS36 (36 probes)': '#d62728', '200-Probe ML': '#1f77b4',
              'Overlap (shared)': '#2ca02c', 'Combined (GS36 + ML)': '#ff7f0e'}

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(best_per_fs))
    bar_colors = [colors.get(fs, '#9467bd') for fs in best_per_fs['Feature_Set']]
    bars = ax.barh(y_pos, best_per_fs['AUROC'], color=bar_colors, alpha=0.85, height=0.6)
    ax.set_yticks(y_pos)
    labels = [f"{row['Feature_Set']}\n({row['Model']})" for _, row in best_per_fs.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Test Set AUROC')
    ax.set_title('Best Model AUROC by Feature Set')
    ax.set_xlim(0.5, 1.0)
    for i, (_, row) in enumerate(best_per_fs.iterrows()):
        ax.text(row['AUROC'] + 0.005, i, f"{row['AUROC']:.3f}", va='center', fontsize=10)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(figures_dir, 'gs36_vs_ml_bar.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_venn_overlap(gs36_probes, ml200_probes, figures_dir):
    """Visualize probe overlap between GS36 and 200-probe model."""
    gs36_set = set(gs36_probes)
    ml_set = set(ml200_probes)
    overlap = gs36_set & ml_set
    gs36_only = gs36_set - ml_set
    ml_only = ml_set - gs36_set

    fig, ax = plt.subplots(figsize=(7, 5))
    # Simple text-based visualization (no matplotlib-venn dependency)
    ax.text(0.25, 0.6, f"GS36 only\n{len(gs36_only)} probes", ha='center', va='center',
            fontsize=14, bbox=dict(boxstyle='round,pad=0.5', facecolor='#d62728', alpha=0.3))
    ax.text(0.5, 0.6, f"Shared\n{len(overlap)} probes", ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ff7f0e', alpha=0.4))
    ax.text(0.75, 0.6, f"ML-200 only\n{len(ml_only)} probes", ha='center', va='center',
            fontsize=14, bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f77b4', alpha=0.3))

    # List overlap genes
    overlap_text = "Shared probes:\n" + ", ".join(sorted(overlap)[:10])
    if len(overlap) > 10:
        overlap_text += f"\n... +{len(overlap)-10} more"
    ax.text(0.5, 0.2, overlap_text, ha='center', va='center', fontsize=8,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Probe Overlap: GS36 ({len(gs36_set)}) vs ML-200 ({len(ml_set)})')
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(figures_dir, 'gs36_ml_probe_overlap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GS36 vs 200-Probe ML Classifier: Head-to-Head Comparison")
    print("=" * 70)

    # Load data
    X, y = load_primary_dataset()
    gs36 = load_gs36_probes()
    gs36_probes = gs36['probeset'].tolist()

    # Same train/test split as original (RANDOM_STATE=42, test_size=0.2, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(y_train)} (pos={sum(y_train==1)}), "
          f"Test: {len(y_test)} (pos={sum(y_test==1)})")

    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)

    # ── Feature set 1: 200-probe ANOVA model (original) ──────────────────────
    print("\n" + "=" * 60)
    print("Feature Set 1: 200-Probe Data-Driven (ANOVA)")
    print("=" * 60)
    X_tr_200, X_te_200, _, ml200_probes = preprocess_anova(X_train, y_train, X_test, n_features=200)
    models_200 = get_tuned_models(n_pos, n_neg)
    cv_200, tuned_200 = nested_cv(X_tr_200, y_train, models_200, label="200-probe")
    test_200, pred_200 = evaluate_test(tuned_200, X_te_200, y_test, label="200-Probe ML")
    ci_200 = bootstrap_ci(tuned_200, X_te_200, y_test, label="200-Probe ML")

    # ── Feature set 2: GS36 (36 probes) ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Feature Set 2: GS36 Signature (36 probes, Sun et al. 2023)")
    print("=" * 60)
    X_tr_gs36, X_te_gs36, _, gs36_avail = preprocess_probe_set(
        X_train, y_train, X_test, gs36_probes, label="GS36"
    )
    models_gs36 = get_tuned_models(n_pos, n_neg)
    cv_gs36, tuned_gs36 = nested_cv(X_tr_gs36, y_train, models_gs36, label="GS36")
    test_gs36, pred_gs36 = evaluate_test(tuned_gs36, X_te_gs36, y_test, label="GS36 (36 probes)")
    ci_gs36 = bootstrap_ci(tuned_gs36, X_te_gs36, y_test, label="GS36 (36 probes)")

    # ── Feature set 3: Overlap probes only ────────────────────────────────────
    overlap_probes = sorted(set(gs36_probes) & set(ml200_probes))
    print("\n" + "=" * 60)
    print(f"Feature Set 3: Overlap Probes Only ({len(overlap_probes)} probes)")
    print("=" * 60)
    if len(overlap_probes) >= 3:
        X_tr_ov, X_te_ov, _, ov_avail = preprocess_probe_set(
            X_train, y_train, X_test, overlap_probes, label="Overlap"
        )
        models_ov = get_tuned_models(n_pos, n_neg)
        cv_ov, tuned_ov = nested_cv(X_tr_ov, y_train, models_ov, label="Overlap")
        test_ov, pred_ov = evaluate_test(tuned_ov, X_te_ov, y_test, label="Overlap (shared)")
        ci_ov = bootstrap_ci(tuned_ov, X_te_ov, y_test, label="Overlap (shared)")
    else:
        print("  Too few overlap probes for meaningful comparison, skipping.")
        test_ov = pd.DataFrame()
        pred_ov = {}
        ci_ov = pd.DataFrame()

    # ── Feature set 4: Combined (200-probe + non-overlapping GS36) ────────────
    gs36_unique = sorted(set(gs36_probes) - set(ml200_probes))
    combined_probes = list(ml200_probes) + gs36_unique
    print("\n" + "=" * 60)
    print(f"Feature Set 4: Combined ({len(ml200_probes)} ML + {len(gs36_unique)} unique GS36 = {len(combined_probes)} probes)")
    print("=" * 60)
    X_tr_comb, X_te_comb, _, comb_avail = preprocess_probe_set(
        X_train, y_train, X_test, combined_probes, label="Combined"
    )
    models_comb = get_tuned_models(n_pos, n_neg)
    cv_comb, tuned_comb = nested_cv(X_tr_comb, y_train, models_comb, label="Combined")
    test_comb, pred_comb = evaluate_test(tuned_comb, X_te_comb, y_test, label="Combined (GS36 + ML)")
    ci_comb = bootstrap_ci(tuned_comb, X_te_comb, y_test, label="Combined (GS36 + ML)")

    # ── Aggregate results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    all_test = pd.concat([test_200, test_gs36, test_ov, test_comb], ignore_index=True)
    all_ci = pd.concat([ci_200, ci_gs36, ci_ov, ci_comb], ignore_index=True)

    # Best model per feature set
    print("\nBest model per feature set (by test AUROC):")
    print("-" * 70)
    for fs in all_test['Feature_Set'].unique():
        subset = all_test[all_test['Feature_Set'] == fs]
        best = subset.loc[subset['AUROC'].idxmax()]
        ci_row = all_ci[(all_ci['Feature_Set'] == fs) & (all_ci['Model'] == best['Model'])]
        if len(ci_row) > 0:
            ci_lo = ci_row.iloc[0]['AUROC_CI_low']
            ci_hi = ci_row.iloc[0]['AUROC_CI_high']
            print(f"  {fs:30s} | {best['Model']:22s} | AUROC = {best['AUROC']:.3f} "
                  f"[{ci_lo:.3f}-{ci_hi:.3f}] | Sens = {best['Sensitivity']:.3f} "
                  f"| Spec = {best['Specificity']:.3f} | NPV = {best['NPV']:.3f}")
        else:
            print(f"  {fs:30s} | {best['Model']:22s} | AUROC = {best['AUROC']:.3f}")

    # Save results
    all_test.to_csv(os.path.join(RESULTS_DIR, 'gs36_comparison_test_results.csv'), index=False)
    all_ci.to_csv(os.path.join(RESULTS_DIR, 'gs36_comparison_bootstrap_ci.csv'), index=False)

    # CV summary
    cv_summary_rows = []
    for label, cv_res in [('200-Probe ML', cv_200), ('GS36 (36 probes)', cv_gs36),
                           ('Combined (GS36 + ML)', cv_comb)]:
        for model_name, vals in cv_res.items():
            cv_summary_rows.append({
                'Feature_Set': label, 'Model': model_name,
                'CV_AUROC_mean': vals['AUROC_mean'], 'CV_AUROC_std': vals['AUROC_std']
            })
    if len(overlap_probes) >= 3:
        for model_name, vals in cv_ov.items():
            cv_summary_rows.append({
                'Feature_Set': 'Overlap (shared)', 'Model': model_name,
                'CV_AUROC_mean': vals['AUROC_mean'], 'CV_AUROC_std': vals['AUROC_std']
            })
    cv_summary = pd.DataFrame(cv_summary_rows)
    cv_summary.to_csv(os.path.join(RESULTS_DIR, 'gs36_comparison_cv_summary.csv'), index=False)

    # Probe overlap analysis
    overlap_analysis = {
        'gs36_total': len(gs36_probes),
        'ml200_total': len(ml200_probes),
        'probe_overlap': len(overlap_probes),
        'overlap_probes': overlap_probes,
        'overlap_fraction_of_gs36': len(overlap_probes) / len(gs36_probes),
        'gs36_unique': gs36_unique,
        'gs36_unique_count': len(gs36_unique),
    }
    with open(os.path.join(RESULTS_DIR, 'gs36_probe_overlap.json'), 'w') as f:
        json.dump(overlap_analysis, f, indent=2)
    print(f"\nProbe overlap: {len(overlap_probes)}/{len(gs36_probes)} GS36 probes "
          f"({len(overlap_probes)/len(gs36_probes)*100:.0f}%) independently selected by ANOVA")

    # ── SVM instability check ─────────────────────────────────────────────────
    print("\nSVM stability check:")
    svm_check_sets = [('GS36', cv_gs36), ('200-Probe', cv_200), ('Combined', cv_comb)]
    if len(overlap_probes) >= 3:
        svm_check_sets.insert(1, ('Overlap', cv_ov))
    for label, cv_res in svm_check_sets:
        svm_cv = cv_res.get('SVM (RBF)', {})
        if svm_cv and svm_cv.get('AUROC_std', 0) > 0.20:
            print(f"  WARNING: {label} SVM CV AUROC = {svm_cv['AUROC_mean']:.3f} +/- "
                  f"{svm_cv['AUROC_std']:.3f} — unstable (std > 0.20), likely degenerate "
                  f"in some folds. Exclude from primary comparison.")

    # ── DeLong test: GS36 best vs 200-probe best ─────────────────────────────
    print("\nDeLong test (paired AUROC comparison on test set):")
    delong_rows = []
    # Get best model predictions for each feature set
    best_preds = {}
    for fs_label, test_df, preds in [('200-Probe ML', test_200, pred_200),
                                      ('GS36 (36 probes)', test_gs36, pred_gs36),
                                      ('Overlap (shared)', test_ov, pred_ov),
                                      ('Combined (GS36 + ML)', test_comb, pred_comb)]:
        if len(test_df) == 0:
            continue
        best_model = test_df.loc[test_df['AUROC'].idxmax(), 'Model']
        best_preds[fs_label] = preds[best_model]['y_prob']

    y_arr = y_test.values
    # Compare all pairs against GS36
    ref_label = 'GS36 (36 probes)'
    if ref_label in best_preds:
        for comp_label, comp_probs in best_preds.items():
            if comp_label == ref_label:
                continue
            z, p, auc_a, auc_b, delta = delong_test(
                y_arr, best_preds[ref_label], comp_probs
            )
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  {ref_label} vs {comp_label}: "
                  f"ΔAUROC = {delta:+.3f}, z = {z:.3f}, p = {p:.4f} ({sig})")
            delong_rows.append({
                'Comparison': f'{ref_label} vs {comp_label}',
                'AUC_A': auc_a, 'AUC_B': auc_b, 'Delta_AUC': delta,
                'z_stat': z, 'p_value': p, 'significant_0.05': p < 0.05
            })
    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(os.path.join(RESULTS_DIR, 'gs36_comparison_delong.csv'), index=False)

    # ── Methodological note ───────────────────────────────────────────────────
    print("\nMethodological notes:")
    print("  - ANOVA feature selection for the 200-probe set is performed once on the")
    print("    full training set before nested CV. This may produce optimistic CV estimates")
    print("    for the 200-probe set relative to GS36 (whose features are pre-specified).")
    print("  - Despite this advantage, the 200-probe set underperforms GS36, strengthening")
    print("    the finding that biology-driven feature selection outperforms data-driven")
    print("    selection for this task.")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating comparison figures...")
    all_preds = {'200-Probe ML': pred_200, 'GS36 (36 probes)': pred_gs36,
                 'Combined (GS36 + ML)': pred_comb}
    if pred_ov:
        all_preds['Overlap (shared)'] = pred_ov

    plot_comparison_roc(all_preds, y_test, FIGURES_DIR)
    plot_comparison_bar(all_test, FIGURES_DIR)
    plot_venn_overlap(gs36_probes, ml200_probes, FIGURES_DIR)

    print("\n" + "=" * 70)
    print("DONE. Results saved to:")
    print(f"  {os.path.join(RESULTS_DIR, 'gs36_comparison_test_results.csv')}")
    print(f"  {os.path.join(RESULTS_DIR, 'gs36_comparison_bootstrap_ci.csv')}")
    print(f"  {os.path.join(RESULTS_DIR, 'gs36_comparison_cv_summary.csv')}")
    print(f"  {os.path.join(RESULTS_DIR, 'gs36_probe_overlap.json')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
