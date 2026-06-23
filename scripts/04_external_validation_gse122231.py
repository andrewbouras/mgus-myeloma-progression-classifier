#!/usr/bin/env python3
"""External Validation of MGUS Progression Classifier on GSE122231 (SWOG S0120).

GSE122231: 216 CD138+ bone marrow plasma cell samples from SWOG S0120
  - GPL570 platform (identical to training data GSE235356)
  - Patient groups: MGUS, AMM (asymptomatic/smouldering MM), WM, MM
  - Molecular subgroups: MF, CD-2, LB, MS, HY, PR, CD-1
  - Timepoints: Baseline, 12 Mo Follow-Up

Validation strategy:
  A. Probe overlap verification (200 probes, both GPL570)
  B. Retrain SVM-RBF + LogisticRegression on full GSE235356 (358 samples)
  C. Apply models to GSE122231 (216 samples)
  D. Statistical analysis (group comparisons, batch effects, distribution shift)
  E. MPS scoring from mps_module_weights.json
  F. Figures (300 DPI PNG)
"""

import os
import sys
import json
import gzip
import warnings
import urllib.request
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # logistic sigmoid

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------- Paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'analysis', 'results')
SCORING_DIR = os.path.join(PROJECT_DIR, 'scoring_system')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plot style
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})


# ============================================================
# DATA DOWNLOAD / LOADING
# ============================================================

def download_gse235356_if_needed():
    """Download and parse GSE235356 if not present locally."""
    expr_path = os.path.join(DATA_DIR, 'GSE235356_expression.csv')
    pheno_path = os.path.join(DATA_DIR, 'GSE235356_phenotype.csv')

    if os.path.exists(expr_path) and os.path.exists(pheno_path):
        print("  GSE235356 data already exists locally.")
        return

    gz_path = os.path.join(DATA_DIR, 'GSE235356_series_matrix.txt.gz')
    if not os.path.exists(gz_path):
        gse_id = 'GSE235356'
        prefix = gse_id[:len(gse_id)-3] + 'nnn'
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
        print(f"  Downloading {url}...")
        urllib.request.urlretrieve(url, gz_path)
        print(f"  Saved ({os.path.getsize(gz_path)/1024/1024:.1f} MB)")

    print("  Parsing GSE235356 series matrix...")
    metadata = {}
    expression_lines = []
    in_table = False

    with gzip.open(gz_path, 'rt', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('!series_matrix_table_begin'):
                in_table = True
                continue
            if line.startswith('!series_matrix_table_end'):
                in_table = False
                continue
            if in_table:
                expression_lines.append(line)
            elif line.startswith('!Sample_'):
                key = line.split('\t')[0].replace('!', '')
                values = line.split('\t')[1:]
                values = [v.strip('"') for v in values]
                metadata.setdefault(key, []).append(values)

    # Expression matrix
    header = expression_lines[0].split('\t')
    sample_ids = [h.strip('"') for h in header[1:]]
    rows = []
    probe_ids = []
    for line in expression_lines[1:]:
        parts = line.split('\t')
        probe_ids.append(parts[0].strip('"'))
        row = []
        for v in parts[1:]:
            try:
                row.append(float(v))
            except (ValueError, IndexError):
                row.append(np.nan)
        rows.append(row)
    expr = pd.DataFrame(rows, index=probe_ids, columns=sample_ids)
    expr.index.name = 'ID_REF'

    # Phenotype table
    pheno_dict = {}
    for key, value_lists in metadata.items():
        if len(value_lists) == 1:
            pheno_dict[key] = value_lists[0]
        else:
            for i, vl in enumerate(value_lists):
                pheno_dict[f'{key}_{i}'] = vl

    if 'Sample_geo_accession' in pheno_dict:
        pheno_index = pheno_dict['Sample_geo_accession']
    else:
        pheno_index = sample_ids

    pheno = pd.DataFrame(pheno_dict, index=pheno_index)

    # Extract characteristics
    char_cols = [c for c in pheno.columns if 'characteristics' in c.lower()]
    for col in char_cols:
        for idx, val in pheno[col].items():
            if ':' in str(val):
                key_part, val_part = str(val).split(':', 1)
                key_part = key_part.strip()
                val_part = val_part.strip()
                col_name = f'char_{key_part}'
                pheno.loc[idx, col_name] = val_part

    expr.to_csv(expr_path)
    pheno.to_csv(pheno_path)
    print(f"  Saved: {expr.shape[0]} probes x {expr.shape[1]} samples")


def load_training_data():
    """Load GSE235356 expression and phenotype."""
    print("\nLoading GSE235356 training data...")
    download_gse235356_if_needed()

    expr = pd.read_csv(os.path.join(DATA_DIR, 'GSE235356_expression.csv'), index_col=0)
    pheno = pd.read_csv(os.path.join(DATA_DIR, 'GSE235356_phenotype.csv'), index_col=0)

    # Identify the disease status column
    # From 02_ml_classifier.py: pheno['char_disease state'] == 'Progressing MGUS'
    target_col = None
    for col in pheno.columns:
        if 'disease' in col.lower() and 'state' in col.lower():
            target_col = col
            break
        elif 'disease_status' in col.lower():
            target_col = col
            break
        elif 'disease state' in col.lower():
            target_col = col
            break

    if target_col is None:
        # Fallback: search for any column containing 'Progressing'
        for col in pheno.columns:
            vals = pheno[col].astype(str).unique()
            if any('rogress' in str(v) for v in vals):
                target_col = col
                break

    if target_col is None:
        raise ValueError("Cannot find disease status column in GSE235356 phenotype!")

    print(f"  Target column: '{target_col}'")
    print(f"  Values: {pheno[target_col].value_counts().to_dict()}")

    y = (pheno[target_col] == 'Progressing MGUS').astype(int)
    X = expr.T
    X.index = X.index.astype(str)
    y.index = y.index.astype(str)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    print(f"  {X.shape[0]} samples, {X.shape[1]} probes")
    print(f"  Stable={sum(y==0)}, Progressing={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    return X, y


def load_validation_data():
    """Load GSE122231 expression and phenotype."""
    print("\nLoading GSE122231 validation data...")
    expr = pd.read_csv(os.path.join(DATA_DIR, 'GSE122231_expression.csv'), index_col=0)
    pheno = pd.read_csv(os.path.join(DATA_DIR, 'GSE122231_phenotype.csv'), index_col=0)

    X_val = expr.T
    X_val.index = X_val.index.astype(str)
    pheno.index = pheno.index.astype(str)

    common = X_val.index.intersection(pheno.index)
    X_val = X_val.loc[common]
    pheno = pheno.loc[common]

    print(f"  {X_val.shape[0]} samples, {X_val.shape[1]} probes")
    print(f"  Patient groups: {pheno['char_patient_group'].value_counts().to_dict()}")
    print(f"  Molecular subgroups: {pheno['char_molecular_subgroup'].value_counts().to_dict()}")
    print(f"  Timepoints: {pheno['char_patient_disease_timepoint'].value_counts().to_dict()}")
    return X_val, pheno


# ============================================================
# A. PROBE OVERLAP VERIFICATION
# ============================================================

def verify_probe_overlap(X_train, X_val, feature_probes):
    """Confirm all 200 probes from feature_importance.csv are present in both datasets."""
    print("\n" + "=" * 60)
    print("A. PROBE OVERLAP VERIFICATION")
    print("=" * 60)

    train_probes = set(X_train.columns)
    val_probes = set(X_val.columns)
    feature_set = set(feature_probes)

    in_train = feature_set.intersection(train_probes)
    in_val = feature_set.intersection(val_probes)
    in_both = in_train.intersection(in_val)
    missing_train = feature_set - train_probes
    missing_val = feature_set - val_probes

    print(f"  Feature probes: {len(feature_probes)}")
    print(f"  In training (GSE235356): {len(in_train)}/{len(feature_probes)}")
    print(f"  In validation (GSE122231): {len(in_val)}/{len(feature_probes)}")
    print(f"  In both datasets: {len(in_both)}/{len(feature_probes)}")

    if missing_train:
        print(f"  Missing from training: {missing_train}")
    if missing_val:
        print(f"  Missing from validation: {missing_val}")

    if len(in_both) == len(feature_probes):
        print("  PASS: All 200 probes present in both GPL570 datasets.")
    else:
        print(f"  WARNING: {len(feature_probes) - len(in_both)} probes missing. Using {len(in_both)} available.")

    usable_probes = sorted(list(in_both))
    return usable_probes


# ============================================================
# B. RETRAIN ON FULL GSE235356
# ============================================================

def train_full_models(X_train, y_train, probes):
    """Train SVM-RBF and LogisticRegression on all 358 training samples."""
    print("\n" + "=" * 60)
    print("B. RETRAIN ON FULL GSE235356")
    print("=" * 60)

    X = X_train[probes].copy()

    # Handle NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"  Imputing {nan_count} NaN values with column medians...")
        X = X.fillna(X.median())

    # StandardScaler fit on training data
    scaler_train = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler_train.fit_transform(X),
        index=X.index,
        columns=X.columns
    )

    print(f"  Training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} probes")
    print(f"  Class balance: Stable={sum(y_train==0)}, Progressing={sum(y_train==1)}")

    # SVM-RBF
    print("\n  Training SVM-RBF (C=10.0, gamma=0.001)...")
    svm_model = SVC(
        C=10.0,
        gamma=0.001,
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    svm_model.fit(X_scaled, y_train)
    train_svm_auc = roc_auc_score(y_train, svm_model.predict_proba(X_scaled)[:, 1])
    print(f"    Training AUC: {train_svm_auc:.4f}")

    # Logistic Regression
    print("\n  Training Logistic Regression (C=1.0)...")
    lr_model = LogisticRegression(
        C=1.0,
        penalty='l2',
        class_weight='balanced',
        solver='lbfgs',
        max_iter=5000,
        random_state=RANDOM_STATE
    )
    lr_model.fit(X_scaled, y_train)
    train_lr_auc = roc_auc_score(y_train, lr_model.predict_proba(X_scaled)[:, 1])
    print(f"    Training AUC: {train_lr_auc:.4f}")

    return svm_model, lr_model, scaler_train


# ============================================================
# C. APPLY TO GSE122231
# ============================================================

def apply_to_validation(X_val, probes, svm_model, lr_model, scaler_train):
    """Apply trained models to GSE122231 validation data.

    Per specification: StandardScaler fit to GSE122231's own distribution
    (not the training scaler). This addresses cross-dataset batch effects
    when both platforms are identical (GPL570) but processed in different labs.
    """
    print("\n" + "=" * 60)
    print("C. APPLY TO GSE122231")
    print("=" * 60)

    X = X_val[probes].copy()

    # Handle NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"  Imputing {nan_count} NaN values with column medians...")
        X = X.fillna(X.median())

    # Scale using validation-fitted scaler (fit to GSE122231's own distribution)
    # This normalizes each probe to zero mean / unit variance within the
    # validation cohort, matching the z-score space the SVM was trained in.
    scaler_val = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler_val.fit_transform(X),
        index=X.index,
        columns=X.columns
    )
    print(f"  StandardScaler fit to GSE122231 distribution (NOT training scaler)")

    print(f"  Validation data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} probes")

    # SVM predictions
    svm_proba = svm_model.predict_proba(X_scaled)[:, 1]
    svm_pred = svm_model.predict(X_scaled)

    # LR predictions
    lr_proba = lr_model.predict_proba(X_scaled)[:, 1]
    lr_pred = lr_model.predict(X_scaled)

    print(f"\n  SVM-RBF predictions:")
    print(f"    Mean score: {svm_proba.mean():.4f} (SD: {svm_proba.std():.4f})")
    print(f"    Predicted positive: {svm_pred.sum()}/{len(svm_pred)} ({svm_pred.sum()/len(svm_pred)*100:.1f}%)")

    print(f"\n  Logistic Regression predictions:")
    print(f"    Mean score: {lr_proba.mean():.4f} (SD: {lr_proba.std():.4f})")
    print(f"    Predicted positive: {lr_pred.sum()}/{len(lr_pred)} ({lr_pred.sum()/len(lr_pred)*100:.1f}%)")

    return X_scaled, svm_proba, svm_pred, lr_proba, lr_pred


# ============================================================
# D. STATISTICAL ANALYSIS
# ============================================================

def analyze_scores(pheno, svm_proba, lr_proba, X_train_scaled, X_val_scaled, y_train):
    """Comprehensive statistical analysis of prediction scores."""
    print("\n" + "=" * 60)
    print("D. STATISTICAL ANALYSIS")
    print("=" * 60)

    results = {}

    # --- D1: Score distribution summary ---
    print("\n  D1. Score Distribution Summary")
    for group_name in pheno['char_patient_group'].unique():
        mask = pheno['char_patient_group'] == group_name
        n = mask.sum()
        scores = svm_proba[mask]
        print(f"    {group_name} (n={n}): mean={scores.mean():.4f}, median={np.median(scores):.4f}, "
              f"SD={scores.std():.4f}, range=[{scores.min():.4f}, {scores.max():.4f}]")

    # --- D2: MGUS vs AMM/WM/MM comparison ---
    print("\n  D2. MGUS vs Non-MGUS (AMM/WM/MM) Comparison")
    is_mgus = pheno['char_patient_group'] == 'MGUS'
    is_non_mgus = pheno['char_patient_group'].isin(['AMM', 'WM', 'MM'])

    mgus_scores = svm_proba[is_mgus]
    non_mgus_scores = svm_proba[is_non_mgus]

    # Mann-Whitney U test
    u_stat, p_mw = stats.mannwhitneyu(mgus_scores, non_mgus_scores, alternative='two-sided')
    # Effect size (rank-biserial correlation)
    n1, n2 = len(mgus_scores), len(non_mgus_scores)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)  # rank-biserial correlation
    # Cohen's d
    pooled_sd = np.sqrt(((n1-1)*mgus_scores.std()**2 + (n2-1)*non_mgus_scores.std()**2) / (n1+n2-2))
    cohens_d = (non_mgus_scores.mean() - mgus_scores.mean()) / pooled_sd if pooled_sd > 0 else 0

    print(f"    MGUS (n={n1}): mean={mgus_scores.mean():.4f}")
    print(f"    Non-MGUS (n={n2}): mean={non_mgus_scores.mean():.4f}")
    print(f"    Mann-Whitney U = {u_stat:.1f}, p = {p_mw:.2e}")
    print(f"    Rank-biserial r = {r_rb:.4f}")
    print(f"    Cohen's d = {cohens_d:.4f}")

    results['mgus_vs_nonmgus'] = {
        'U': float(u_stat), 'p': float(p_mw),
        'rank_biserial_r': float(r_rb), 'cohens_d': float(cohens_d)
    }

    # AUROC: MGUS (0) vs non-MGUS (1)
    binary_labels = (~is_mgus).astype(int).values
    # Only for samples that are MGUS or non-MGUS
    valid_mask = is_mgus | is_non_mgus
    if valid_mask.sum() > 0 and len(np.unique(binary_labels[valid_mask])) == 2:
        auroc = roc_auc_score(binary_labels[valid_mask], svm_proba[valid_mask])
        print(f"    AUROC (MGUS vs non-MGUS): {auroc:.4f}")
        results['mgus_vs_nonmgus']['auroc'] = float(auroc)

    # --- D3: By molecular subgroup (Kruskal-Wallis + pairwise) ---
    print("\n  D3. By Molecular Subgroup")
    subgroups = pheno['char_molecular_subgroup'].dropna().unique()
    group_scores = {}
    for sg in sorted(subgroups):
        mask = pheno['char_molecular_subgroup'] == sg
        group_scores[sg] = svm_proba[mask]
        print(f"    {sg} (n={mask.sum()}): mean={svm_proba[mask].mean():.4f}, SD={svm_proba[mask].std():.4f}")

    if len(group_scores) >= 2:
        score_arrays = [v for v in group_scores.values() if len(v) >= 2]
        h_stat, p_kw = stats.kruskal(*score_arrays)
        print(f"\n    Kruskal-Wallis H = {h_stat:.2f}, p = {p_kw:.2e}")
        results['molecular_subgroup_kruskal'] = {'H': float(h_stat), 'p': float(p_kw)}

        # Pairwise Mann-Whitney with Bonferroni correction
        print("\n    Pairwise Mann-Whitney U (Bonferroni-corrected):")
        sg_names = sorted(group_scores.keys())
        n_comparisons = len(sg_names) * (len(sg_names) - 1) // 2
        pairwise_results = []
        for i in range(len(sg_names)):
            for j in range(i+1, len(sg_names)):
                g1, g2 = sg_names[i], sg_names[j]
                if len(group_scores[g1]) >= 2 and len(group_scores[g2]) >= 2:
                    u, p = stats.mannwhitneyu(group_scores[g1], group_scores[g2], alternative='two-sided')
                    p_adj = min(p * n_comparisons, 1.0)
                    sig = '*' if p_adj < 0.05 else ''
                    print(f"      {g1} vs {g2}: U={u:.0f}, p_adj={p_adj:.4f} {sig}")
                    pairwise_results.append({'g1': g1, 'g2': g2, 'U': float(u), 'p_adj': float(p_adj)})
        results['pairwise_subgroups'] = pairwise_results

    # --- D4: Batch effect assessment (PCA) ---
    print("\n  D4. Batch Effect Assessment")

    # Use the same 200 probes from both datasets (already scaled)
    # Re-create combined matrix from raw (pre-scaled) for PCA
    # We need the raw expression for PCA visualization
    combined = pd.concat([X_train_scaled, X_val_scaled], axis=0)
    dataset_labels = ['GSE235356'] * len(X_train_scaled) + ['GSE122231'] * len(X_val_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(combined.values)

    print(f"    PCA on combined data ({combined.shape[0]} samples, {combined.shape[1]} probes)")
    print(f"    PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")
    print(f"    PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% variance")

    results['pca'] = {
        'pc1_var': float(pca.explained_variance_ratio_[0]),
        'pc2_var': float(pca.explained_variance_ratio_[1]),
    }

    # --- D5: Distribution shift (KS test) ---
    print("\n  D5. Distribution Shift Assessment")

    # Get training SVM scores for comparison
    svm_train_proba = svm_proba  # We'll compute this in the caller
    # Actually we need training scores -- pass them in. For now, use the
    # decision function approach: recompute from the model
    # Note: training scores already computed in main()
    # We'll handle this via return values

    # --- D6: By timepoint ---
    print("\n  D6. By Timepoint")
    timepoints = pheno['char_patient_disease_timepoint'].dropna().unique()
    tp_scores = {}
    for tp in sorted(timepoints):
        mask = pheno['char_patient_disease_timepoint'] == tp
        tp_scores[tp] = svm_proba[mask]
        print(f"    {tp} (n={mask.sum()}): mean={svm_proba[mask].mean():.4f}, "
              f"SD={svm_proba[mask].std():.4f}")

    if len(tp_scores) == 2:
        tp_names = sorted(tp_scores.keys())
        u, p = stats.mannwhitneyu(tp_scores[tp_names[0]], tp_scores[tp_names[1]], alternative='two-sided')
        print(f"    {tp_names[0]} vs {tp_names[1]}: Mann-Whitney U={u:.0f}, p={p:.4f}")
        results['timepoint_comparison'] = {
            'U': float(u), 'p': float(p),
            'groups': {tp: float(tp_scores[tp].mean()) for tp in tp_names}
        }

    return results, pca_coords, dataset_labels


# ============================================================
# E. MPS SCORING
# ============================================================

def compute_mps_scores(X_val, pheno):
    """Compute MPS scores using saved module weights from mps_module_weights.json."""
    print("\n" + "=" * 60)
    print("E. MPS SCORING (from mps_module_weights.json)")
    print("=" * 60)

    weights_path = os.path.join(SCORING_DIR, 'mps_module_weights.json')
    with open(weights_path, 'r') as f:
        mps_config = json.load(f)

    module_gene_weights = mps_config['module_gene_weights']
    composite_weights = mps_config['composite_module_weights']
    norm_params = mps_config['normalization_params']
    risk_thresholds = mps_config['risk_thresholds']

    # Identify all MPS probes
    all_mps_probes = {}
    for mod_name, genes in module_gene_weights.items():
        for gene_name, gene_info in genes.items():
            all_mps_probes[gene_info['probe']] = (mod_name, gene_name)

    print(f"  Total MPS genes: {len(all_mps_probes)}")

    # Check probe availability
    available = set(X_val.columns)
    present = set(all_mps_probes.keys()).intersection(available)
    missing = set(all_mps_probes.keys()) - available
    print(f"  Probes available in GSE122231: {len(present)}/{len(all_mps_probes)}")
    if missing:
        for p in missing:
            mod, gene = all_mps_probes[p]
            print(f"    MISSING: {gene} ({p}) in {mod}")

    # Scale the MPS probes using StandardScaler (fit to validation data)
    mps_probe_list = sorted(list(present))
    X_mps = X_val[mps_probe_list].copy()
    nan_count = X_mps.isna().sum().sum()
    if nan_count > 0:
        print(f"  Imputing {nan_count} NaN values...")
        X_mps = X_mps.fillna(X_mps.median())

    scaler_mps = StandardScaler()
    X_mps_scaled = pd.DataFrame(
        scaler_mps.fit_transform(X_mps),
        index=X_mps.index,
        columns=X_mps.columns
    )

    # Compute module scores (weighted sum of scaled gene expressions)
    print("\n  Computing module scores...")
    module_scores = pd.DataFrame(index=X_val.index)

    for mod_name, genes in module_gene_weights.items():
        probes = []
        signed_weights = []
        for gene_name, gene_info in genes.items():
            probe = gene_info['probe']
            if probe in X_mps_scaled.columns:
                probes.append(probe)
                signed_weights.append(gene_info['signed_weight'])

        if len(probes) > 0:
            X_mod = X_mps_scaled[probes].values
            w = np.array(signed_weights)
            raw_score = X_mod @ w
            module_scores[mod_name] = raw_score
            print(f"    {mod_name}: {len(probes)} genes, "
                  f"mean={raw_score.mean():.4f}, SD={raw_score.std():.4f}")

    # Normalize using saved quantile parameters
    print("\n  Normalizing with saved quantile parameters...")
    normed_scores = pd.DataFrame(index=X_val.index)
    for mod_name in module_scores.columns:
        q_low = norm_params[mod_name]['q_low']
        q_high = norm_params[mod_name]['q_high']
        normed_scores[mod_name] = np.clip(
            (module_scores[mod_name] - q_low) / (q_high - q_low + 1e-10), 0, 1
        )
        print(f"    {mod_name}: normed mean={normed_scores[mod_name].mean():.4f}, "
              f"range=[{normed_scores[mod_name].min():.4f}, {normed_scores[mod_name].max():.4f}]")

    # Composite MPS using logistic regression weights
    print("\n  Computing composite MPS...")
    intercept = composite_weights['_intercept']
    logit_sum = np.full(len(X_val), intercept)
    for mod_name in normed_scores.columns:
        coef = composite_weights[mod_name]['coefficient']
        logit_sum += coef * normed_scores[mod_name].values

    # Logistic (sigmoid) to probability, then map to 0-10
    prob = expit(logit_sum)
    mps = prob * 10.0

    print(f"\n  MPS Distribution:")
    print(f"    Mean: {mps.mean():.2f}")
    print(f"    Median: {np.median(mps):.2f}")
    print(f"    SD: {mps.std():.2f}")
    print(f"    Range: [{mps.min():.2f}, {mps.max():.2f}]")

    # Risk tier assignment
    low_upper = risk_thresholds['low_upper']
    intermediate_upper = risk_thresholds['intermediate_upper']

    tiers = []
    for s in mps:
        if s < low_upper:
            tiers.append('Low')
        elif s < intermediate_upper:
            tiers.append('Intermediate')
        else:
            tiers.append('High')

    tier_series = pd.Series(tiers, index=X_val.index)

    print(f"\n  Risk Tier Distribution:")
    tier_counts = tier_series.value_counts()
    for tier in ['Low', 'Intermediate', 'High']:
        n = tier_counts.get(tier, 0)
        pct = n / len(tier_series) * 100
        print(f"    {tier}: {n} ({pct:.1f}%)")

    # Report by patient group
    print("\n  Risk Tiers by Patient Group:")
    for group in sorted(pheno['char_patient_group'].unique()):
        mask = pheno['char_patient_group'] == group
        group_tiers = tier_series[mask]
        counts = group_tiers.value_counts()
        total = mask.sum()
        parts = []
        for tier in ['Low', 'Intermediate', 'High']:
            n = counts.get(tier, 0)
            parts.append(f"{tier}={n}({n/total*100:.0f}%)")
        print(f"    {group} (n={total}): {', '.join(parts)}")

    return mps, tier_series, module_scores, normed_scores


# ============================================================
# F. FIGURES
# ============================================================

def plot_score_distribution_by_group(pheno, svm_proba, lr_proba):
    """Figure 1: Score distribution histogram by patient group."""
    print("\n  Generating Figure 1: Score distribution by patient group...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    groups = pheno['char_patient_group'].values
    group_order = ['MGUS', 'AMM', 'WM', 'MM']
    colors = {'MGUS': '#2196F3', 'AMM': '#FF9800', 'WM': '#4CAF50', 'MM': '#F44336'}

    for ax, (scores, title) in zip(axes, [(svm_proba, 'SVM-RBF'), (lr_proba, 'Logistic Regression')]):
        for group in group_order:
            mask = groups == group
            if mask.sum() > 0:
                ax.hist(scores[mask], bins=25, alpha=0.5, label=f'{group} (n={mask.sum()})',
                        color=colors.get(group, 'gray'), density=True)
                # KDE
                if mask.sum() >= 5:
                    try:
                        kde_x = np.linspace(0, 1, 200)
                        kde = stats.gaussian_kde(scores[mask])
                        ax.plot(kde_x, kde(kde_x), color=colors.get(group, 'gray'), linewidth=2)
                    except Exception:
                        pass

        ax.set_xlabel('Progression Probability Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{title} Score Distribution by Patient Group')
        ax.legend(frameon=True, fontsize=9)
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'gse122231_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved: figures/gse122231_score_distribution.png")


def plot_pca_batch_effect(pca_coords, dataset_labels, y_train, pheno):
    """Figure 2: PCA batch effect assessment."""
    print("\n  Generating Figure 2: PCA batch effect...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = np.array(dataset_labels)

    # Panel A: Colored by dataset
    ax = axes[0]
    for ds, color, marker in [('GSE235356', '#2196F3', 'o'), ('GSE122231', '#F44336', 's')]:
        mask = labels == ds
        ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                   c=color, alpha=0.5, s=20, marker=marker, label=ds, edgecolors='none')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA: Colored by Dataset')
    ax.legend(frameon=True)

    # Panel B: GSE122231 colored by patient group
    ax = axes[1]
    n_train = len(y_train)
    val_coords = pca_coords[n_train:]
    val_pheno = pheno

    group_colors = {'MGUS': '#2196F3', 'AMM': '#FF9800', 'WM': '#4CAF50', 'MM': '#F44336'}
    for group, color in group_colors.items():
        mask = val_pheno['char_patient_group'].values == group
        if mask.sum() > 0:
            ax.scatter(val_coords[mask, 0], val_coords[mask, 1],
                       c=color, alpha=0.6, s=25, label=f'{group} (n={mask.sum()})', edgecolors='none')

    # Also plot training in gray background
    ax.scatter(pca_coords[:n_train, 0], pca_coords[:n_train, 1],
               c='lightgray', alpha=0.2, s=10, label='GSE235356 (background)', edgecolors='none')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA: GSE122231 Patient Groups')
    ax.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'gse122231_pca_batch_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved: figures/gse122231_pca_batch_effect.png")


def plot_mps_risk_tiers(tier_series, pheno):
    """Figure 3: MPS risk tier distribution."""
    print("\n  Generating Figure 3: MPS risk tier distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tier_colors = {'Low': '#4CAF50', 'Intermediate': '#FF9800', 'High': '#F44336'}
    tier_order = ['Low', 'Intermediate', 'High']

    # Panel A: Overall distribution
    ax = axes[0]
    counts = tier_series.value_counts()
    bars = ax.bar(tier_order, [counts.get(t, 0) for t in tier_order],
                  color=[tier_colors[t] for t in tier_order], edgecolor='black', linewidth=0.5)
    for bar, tier in zip(bars, tier_order):
        n = counts.get(tier, 0)
        pct = n / len(tier_series) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{n}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Number of Samples')
    ax.set_title('MPS Risk Tier Distribution (All Samples)')
    ax.set_ylim(0, max(counts.values) * 1.25)

    # Panel B: By patient group (stacked)
    ax = axes[1]
    group_order = ['MGUS', 'AMM', 'WM', 'MM']
    groups_present = [g for g in group_order if g in pheno['char_patient_group'].values]

    # Build contingency
    data = {}
    for tier in tier_order:
        data[tier] = []
        for group in groups_present:
            mask = pheno['char_patient_group'] == group
            n_tier = (tier_series[mask] == tier).sum()
            data[tier].append(n_tier)

    x = np.arange(len(groups_present))
    width = 0.6
    bottom = np.zeros(len(groups_present))

    for tier in tier_order:
        vals = np.array(data[tier])
        ax.bar(x, vals, width, bottom=bottom, label=tier,
               color=tier_colors[tier], edgecolor='black', linewidth=0.5)
        # Add percentage labels
        totals = np.array([sum(pheno['char_patient_group'] == g) for g in groups_present])
        for i, (v, t) in enumerate(zip(vals, totals)):
            if v > 0:
                pct = v / t * 100
                ax.text(x[i], bottom[i] + v/2, f'{pct:.0f}%',
                        ha='center', va='center', fontsize=8, fontweight='bold')
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(groups_present)
    ax.set_ylabel('Number of Samples')
    ax.set_title('MPS Risk Tiers by Patient Group')
    ax.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'gse122231_mps_risk_tiers.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved: figures/gse122231_mps_risk_tiers.png")


def plot_molecular_subgroup_boxplot(pheno, svm_proba):
    """Figure 4: Score by molecular subgroup (box plot)."""
    print("\n  Generating Figure 4: Score by molecular subgroup...")
    fig, ax = plt.subplots(figsize=(10, 6))

    subgroups = pheno['char_molecular_subgroup'].dropna()
    sg_order = sorted(subgroups.unique(), key=lambda x: -svm_proba[subgroups == x].mean())

    data_for_plot = []
    labels_for_plot = []
    for sg in sg_order:
        mask = subgroups == sg
        data_for_plot.append(svm_proba[mask])
        labels_for_plot.append(f'{sg}\n(n={mask.sum()})')

    bp = ax.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True,
                    widths=0.6, showfliers=True)

    # Color by median score
    cmap = plt.cm.RdYlBu_r
    medians = [np.median(d) for d in data_for_plot]
    norm = plt.Normalize(vmin=min(medians), vmax=max(medians))
    for patch, med in zip(bp['boxes'], medians):
        patch.set_facecolor(cmap(norm(med)))
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, data in enumerate(data_for_plot):
        x = np.random.normal(i + 1, 0.06, size=len(data))
        ax.scatter(x, data, alpha=0.4, s=12, c='black', zorder=3)

    ax.set_ylabel('SVM-RBF Progression Probability')
    ax.set_title('Progression Score by Molecular Subgroup (GSE122231)')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'gse122231_molecular_subgroup.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved: figures/gse122231_molecular_subgroup.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("EXTERNAL VALIDATION: GSE122231 (SWOG S0120)")
    print("=" * 60)

    # Load feature importance probes
    fi = pd.read_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'))
    feature_probes = fi['probe_id'].tolist()
    print(f"\nLoaded {len(feature_probes)} feature probes from feature_importance.csv")

    # Load datasets
    X_train, y_train = load_training_data()
    X_val, pheno_val = load_validation_data()

    # A. Probe overlap verification
    usable_probes = verify_probe_overlap(X_train, X_val, feature_probes)

    # B. Retrain on full GSE235356
    svm_model, lr_model, scaler_train = train_full_models(X_train, y_train, usable_probes)

    # C. Apply to GSE122231
    X_val_scaled, svm_proba, svm_pred, lr_proba, lr_pred = apply_to_validation(
        X_val, usable_probes, svm_model, lr_model, scaler_train
    )

    # Compute training scores for distribution shift comparison
    # Use training scaler (from train_full_models) for training predictions
    X_train_200 = X_train[usable_probes].copy()
    X_train_200 = X_train_200.fillna(X_train_200.median())
    X_train_scaled = pd.DataFrame(
        scaler_train.transform(X_train_200),
        index=X_train_200.index,
        columns=X_train_200.columns
    )
    svm_train_proba = svm_model.predict_proba(X_train_scaled)[:, 1]

    # Load full GSE122231 expression for MPS scoring (needs all 54K probes)
    print("\n  Loading full GSE122231 expression for MPS scoring...")
    expr_val_full = pd.read_csv(os.path.join(DATA_DIR, 'GSE122231_expression.csv'), index_col=0)
    X_val_full = expr_val_full.T
    X_val_full.index = X_val_full.index.astype(str)

    # D. Statistical analysis
    results, pca_coords, dataset_labels = analyze_scores(
        pheno_val, svm_proba, lr_proba, X_train_scaled, X_val_scaled, y_train
    )

    # D5: Distribution shift (deferred from analyze_scores)
    print("\n  D5. Distribution Shift (KS test)")
    ks_stat, ks_p = stats.ks_2samp(svm_train_proba, svm_proba)
    print(f"    KS statistic = {ks_stat:.4f}, p = {ks_p:.2e}")
    print(f"    Training: mean={svm_train_proba.mean():.4f}, SD={svm_train_proba.std():.4f}")
    print(f"    Validation: mean={svm_proba.mean():.4f}, SD={svm_proba.std():.4f}")
    results['distribution_shift'] = {
        'ks_statistic': float(ks_stat), 'ks_p': float(ks_p),
        'train_mean': float(svm_train_proba.mean()),
        'train_sd': float(svm_train_proba.std()),
        'val_mean': float(svm_proba.mean()),
        'val_sd': float(svm_proba.std()),
    }

    # E. MPS scoring (uses full expression matrix with all probes)
    mps, tier_series, module_scores, normed_scores = compute_mps_scores(X_val_full, pheno_val)

    # F. Figures
    print("\n" + "=" * 60)
    print("F. GENERATING FIGURES")
    print("=" * 60)

    plot_score_distribution_by_group(pheno_val, svm_proba, lr_proba)
    plot_pca_batch_effect(pca_coords, dataset_labels, y_train, pheno_val)
    plot_mps_risk_tiers(tier_series, pheno_val)
    plot_molecular_subgroup_boxplot(pheno_val, svm_proba)

    # Save all scores
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    scores_df = pd.DataFrame({
        'sample_id': pheno_val.index,
        'patient_group': pheno_val['char_patient_group'].values,
        'molecular_subgroup': pheno_val['char_molecular_subgroup'].values,
        'timepoint': pheno_val['char_patient_disease_timepoint'].values,
        'gep70_score': pheno_val['char_gep70_score'].values,
        'svm_rbf_score': svm_proba,
        'svm_rbf_prediction': svm_pred,
        'lr_score': lr_proba,
        'lr_prediction': lr_pred,
        'mps_score': np.asarray(mps),
        'mps_risk_tier': np.asarray(tier_series),
    })

    # Add module scores
    for col in module_scores.columns:
        scores_df[f'module_{col.replace(" / ", "_").replace(" ", "_").lower()}'] = module_scores[col].values
    for col in normed_scores.columns:
        scores_df[f'module_{col.replace(" / ", "_").replace(" ", "_").lower()}_normed'] = normed_scores[col].values

    out_path = os.path.join(RESULTS_DIR, 'gse122231_validation_scores.csv')
    scores_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Shape: {scores_df.shape}")

    # Save analysis results as JSON
    results_json_path = os.path.join(RESULTS_DIR, 'gse122231_validation_stats.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_json_path}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Probes used: {len(usable_probes)}/{len(feature_probes)}")
    print(f"  Samples scored: {len(svm_proba)}")
    print(f"  SVM-RBF: mean={svm_proba.mean():.4f}, predicted positive={svm_pred.sum()}")
    print(f"  LR: mean={lr_proba.mean():.4f}, predicted positive={lr_pred.sum()}")
    print(f"  MPS: mean={mps.mean():.2f}, SD={mps.std():.2f}")
    print(f"  Risk tiers: {tier_series.value_counts().to_dict()}")
    if 'mgus_vs_nonmgus' in results:
        r = results['mgus_vs_nonmgus']
        print(f"  MGUS vs non-MGUS: p={r['p']:.2e}, d={r['cohens_d']:.3f}")
        if 'auroc' in r:
            print(f"  AUROC (MGUS vs non-MGUS): {r['auroc']:.4f}")
    print(f"  Distribution shift: KS={results['distribution_shift']['ks_statistic']:.4f}, "
          f"p={results['distribution_shift']['ks_p']:.2e}")

    print("\n  Figures saved to:", FIGURES_DIR)
    print("  Done.")


if __name__ == '__main__':
    main()
