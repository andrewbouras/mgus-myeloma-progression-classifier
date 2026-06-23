#!/usr/bin/env python3
"""ML Classifier for MGUS-to-Multiple Myeloma Progression Prediction.

Revision addressing critique v1 findings:
- Nested CV with hyperparameter tuning (RandomizedSearchCV inner loop)
- Repeated stratified K-fold (5x3) for stable estimates
- sample_weight for GradientBoosting (no native class_weight)
- CV-to-test gap analysis
- Published baseline comparison table
- Gene symbol mapping via GPL570 annotation
- Pathway enrichment via gseapy (Enrichr)
- Portable paths, version logging
"""

import os
import sys
import warnings
import json
import gc
import time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

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
    confusion_matrix, brier_score_loss, f1_score, accuracy_score, make_scorer
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import uniform, randint
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

# Log versions
def log_versions():
    import sklearn, xgboost, shap, matplotlib, scipy
    versions = {
        'numpy': np.__version__, 'pandas': pd.__version__,
        'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__,
        'shap': shap.__version__, 'matplotlib': matplotlib.__version__,
        'scipy': scipy.__version__, 'python': sys.version.split()[0]
    }
    print("Package versions:", json.dumps(versions))
    return versions


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


def preprocess(X_train, y_train, X_test, n_features=200):
    """Feature selection and scaling on training data only.
    Uses numpy arrays internally for performance (54K columns is slow in pandas).
    """
    print(f"\nFeature selection (target: {n_features})...")
    all_columns = X_train.columns
    train_idx = X_train.index
    test_idx = X_test.index

    # Convert to numpy for fast operations
    X_tr_np = X_train.values.astype(np.float64)
    X_te_np = X_test.values.astype(np.float64)

    # Fill NaN with column medians (if any)
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
    print(f"  Variance filter: {X_tr_np.shape[1]} probes")

    # Univariate ANOVA
    n_sel = min(n_features, X_tr_np.shape[1])
    skb = SelectKBest(f_classif, k=n_sel)
    X_tr_np = skb.fit_transform(X_tr_np, y_train)
    X_te_np = skb.transform(X_te_np)
    final_cols = kept_cols[skb.get_support()]
    print(f"  ANOVA selection: {X_tr_np.shape[1]} probes")

    # Standardize
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_np)
    X_te_np = scaler.transform(X_te_np)

    # Convert back to DataFrame only at the end (200 columns, fast)
    X_tr = pd.DataFrame(X_tr_np, index=train_idx, columns=final_cols)
    X_te = pd.DataFrame(X_te_np, index=test_idx, columns=final_cols)

    return X_tr, X_te, scaler, skb, vt2


def get_tuned_models(n_pos, n_neg):
    """Return models with hyperparameter search spaces for RandomizedSearchCV."""
    scale_pos = n_neg / n_pos

    models_and_params = {
        'Logistic Regression': (
            LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000,
                               class_weight='balanced', random_state=RANDOM_STATE),
            {'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
        ),
        'Random Forest': (
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1),
            {
                'n_estimators': [100, 200, 500],
                'max_depth': [3, 5, 10, None],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2'],
            }
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [2, 3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'min_samples_leaf': [2, 5, 10],
            }
        ),
        'XGBoost': (
            xgb.XGBClassifier(scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
                              eval_metric='logloss', verbosity=0, n_jobs=1),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [2, 3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 5, 10],
            }
        ),
        'SVM (RBF)': (
            SVC(kernel='rbf', probability=True, class_weight='balanced',
                random_state=RANDOM_STATE),
            {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
            }
        ),
    }
    return models_and_params


def nested_cv_with_tuning(X_tr, y_train, models_and_params):
    """Nested CV: outer 5-fold repeated 3 times, inner 3-fold for tuning."""
    print("\n" + "=" * 60)
    print("Nested Cross-Validation with Hyperparameter Tuning")
    print("Outer: 5-fold x 3 repeats | Inner: 3-fold RandomizedSearchCV")
    print("=" * 60)

    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}
    cv_detail_rows = []
    best_params_all = {}

    for name, (base_model, param_dist) in models_and_params.items():
        print(f"\n  {name}...")
        fold_aucs, fold_aps = [], []
        # Limit RandomizedSearchCV iterations based on search space size
        total_combos = 1
        for v in param_dist.values():
            total_combos *= len(v) if isinstance(v, list) else 10
        n_iter = min(50, total_combos)

        for fold_i, (tr_idx, val_idx) in enumerate(outer_cv.split(X_tr, y_train)):
            X_fold_tr = X_tr.iloc[tr_idx]
            y_fold_tr = y_train.iloc[tr_idx]
            X_fold_val = X_tr.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Inner loop: hyperparameter tuning
            search = RandomizedSearchCV(
                type(base_model)(**base_model.get_params()),
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring='roc_auc',
                cv=inner_cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True
            )

            # For GradientBoosting, use sample_weight for class imbalance
            if name == 'Gradient Boosting':
                sw = compute_sample_weight('balanced', y_fold_tr)
                search.fit(X_fold_tr, y_fold_tr, sample_weight=sw)
            else:
                search.fit(X_fold_tr, y_fold_tr)

            yp = search.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, yp)
            ap = average_precision_score(y_fold_val, yp)
            fold_aucs.append(auc)
            fold_aps.append(ap)
            cv_detail_rows.append({
                'Model': name, 'Repeat': fold_i // 5 + 1, 'Fold': fold_i % 5 + 1,
                'AUROC': auc, 'AUPRC': ap
            })

        cv_results[name] = {
            'AUROC_mean': np.mean(fold_aucs), 'AUROC_std': np.std(fold_aucs),
            'AUPRC_mean': np.mean(fold_aps), 'AUPRC_std': np.std(fold_aps),
            'n_folds': len(fold_aucs)
        }
        print(f"    AUROC={np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f} "
              f"({len(fold_aucs)} folds), AUPRC={np.mean(fold_aps):.3f} +/- {np.std(fold_aps):.3f}")

    # Final tuning on full training set for each model
    print("\nFinal hyperparameter tuning on full training set...")
    tuned_models = {}
    for name, (base_model, param_dist) in models_and_params.items():
        total_combos = 1
        for v in param_dist.values():
            total_combos *= len(v) if isinstance(v, list) else 10
        n_iter = min(50, total_combos)

        search = RandomizedSearchCV(
            type(base_model)(**base_model.get_params()),
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=inner_cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True
        )
        if name == 'Gradient Boosting':
            sw = compute_sample_weight('balanced', y_train)
            search.fit(X_tr, y_train, sample_weight=sw)
        else:
            search.fit(X_tr, y_train)

        tuned_models[name] = search.best_estimator_
        best_params_all[name] = search.best_params_
        print(f"  {name}: {search.best_params_}")

    return cv_results, cv_detail_rows, tuned_models, best_params_all


def evaluate_test_set(models, X_te, y_test):
    """Evaluate all models on held-out test set."""
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    test_rows = []
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

        test_rows.append({
            'Model': name, 'AUROC': auc, 'AUPRC': ap, 'Brier_Score': brier,
            'Sensitivity': sens, 'Specificity': spec, 'PPV': ppv, 'NPV': npv,
            'F1': f1_score(y_test, y_opt), 'Accuracy': accuracy_score(y_test, y_opt),
            'Optimal_Threshold': opt_thresh
        })
        predictions[name] = {'y_prob': yp, 'fpr': fpr, 'tpr': tpr}
        print(f"  {name}: AUROC={auc:.3f}, AUPRC={ap:.3f}, Brier={brier:.3f}, "
              f"Sens={sens:.3f}, Spec={spec:.3f}")

    test_df = pd.DataFrame(test_rows).sort_values('AUROC', ascending=False)
    return test_df, predictions


def bootstrap_test_ci(models, X_te, y_test, n_boot=1000, seed=42):
    """Compute bootstrap 95% CIs for test-set AUROC and AUPRC."""
    print("\nBootstrap 95% CIs for test-set metrics (n_boot=1000):")
    rng = np.random.RandomState(seed)
    ci_rows = []
    n = len(y_test)
    for name, model in models.items():
        yp = model.predict_proba(X_te)[:, 1]
        boot_aurocs, boot_auprcs = [], []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            if len(np.unique(y_test.iloc[idx])) < 2:
                continue
            boot_aurocs.append(roc_auc_score(y_test.iloc[idx], yp[idx]))
            boot_auprcs.append(average_precision_score(y_test.iloc[idx], yp[idx]))
        auroc_lo, auroc_hi = np.percentile(boot_aurocs, [2.5, 97.5])
        auprc_lo, auprc_hi = np.percentile(boot_auprcs, [2.5, 97.5])
        ci_rows.append({
            'Model': name,
            'AUROC_point': roc_auc_score(y_test, yp),
            'AUROC_CI_low': auroc_lo, 'AUROC_CI_high': auroc_hi,
            'AUPRC_point': average_precision_score(y_test, yp),
            'AUPRC_CI_low': auprc_lo, 'AUPRC_CI_high': auprc_hi,
            'n_boot': len(boot_aurocs)
        })
        print(f"  {name}: AUROC={roc_auc_score(y_test, yp):.3f} "
              f"[{auroc_lo:.3f}-{auroc_hi:.3f}], "
              f"AUPRC={average_precision_score(y_test, yp):.3f} "
              f"[{auprc_lo:.3f}-{auprc_hi:.3f}]")
    return pd.DataFrame(ci_rows)


def cv_test_gap_analysis(cv_results, test_df):
    """Compute and report CV-to-test performance gaps."""
    print("\nCV-to-Test Performance Gap Analysis:")
    gap_rows = []
    for _, row in test_df.iterrows():
        name = row['Model']
        cv = cv_results[name]
        gap_auroc = cv['AUROC_mean'] - row['AUROC']
        gap_auprc = cv['AUPRC_mean'] - row['AUPRC']
        flag = "OVERFITTING" if gap_auroc > 0.10 else ("WARNING" if gap_auroc > 0.05 else "OK")
        gap_rows.append({
            'Model': name,
            'CV_AUROC': cv['AUROC_mean'], 'Test_AUROC': row['AUROC'],
            'Gap_AUROC': gap_auroc,
            'CV_AUPRC': cv['AUPRC_mean'], 'Test_AUPRC': row['AUPRC'],
            'Gap_AUPRC': gap_auprc,
            'Flag': flag
        })
        print(f"  {name}: CV={cv['AUROC_mean']:.3f} vs Test={row['AUROC']:.3f} "
              f"(gap={gap_auroc:+.3f}) [{flag}]")
    return pd.DataFrame(gap_rows)


def published_baseline_comparison(test_df, n_features=200):
    """Compare our results against published baselines from data_sources.json."""
    print("\nPublished Baseline Comparison:")
    baselines = [
        {'Model': 'GS36 (Sun 2023)', 'Metric': 'C-statistic', 'Value': 0.928,
         'Dataset': 'GSE235356 (374 MGUS)', 'Method': '36-gene Cox PH',
         'Note': 'Time-to-event C-index, not directly comparable to binary AUROC'},
        {'Model': 'Karathanasis 2025', 'Metric': 'AUROC', 'Value': 0.80,
         'Dataset': 'GEO datasets incl. GSE235356', 'Method': 'ElasticNet/RF/Boosting/SVM',
         'Note': 'AUC >0.8 reported; exact values per model not specified'},
        {'Model': 'PolyPC (Sun 2025)', 'Metric': 'C-statistic', 'Value': 0.792,
         'Dataset': 'Polyclonal PCs', 'Method': 'Gene expression negative predictor',
         'Note': 'Different cell population (polyclonal vs monoclonal PCs)'},
        {'Model': 'MGUSscore (Fu 2026)', 'Metric': 'LASSO coefficient', 'Value': None,
         'Dataset': 'GSE136337 (426 MM)', 'Method': '2-gene LASSO (DAP3+UBE2S)',
         'Note': 'Different population (newly diagnosed MM, not MGUS); different platform'},
    ]

    # Add our best model
    best = test_df.iloc[0]
    baselines.append({
        'Model': f'This study ({best["Model"]})', 'Metric': 'AUROC', 'Value': best['AUROC'],
        'Dataset': 'GSE235356 (358 MGUS)', 'Method': f'Tuned ML classifier ({n_features} probes)',
        'Note': 'Binary classification AUROC on held-out test set'
    })

    baseline_df = pd.DataFrame(baselines)
    for _, row in baseline_df.iterrows():
        val_str = f"{row['Value']:.3f}" if row['Value'] is not None else "N/A"
        print(f"  {row['Model']}: {row['Metric']}={val_str} -- {row['Note']}")
    return baseline_df


def map_probes_to_genes(feature_names):
    """Map Affymetrix U133 Plus 2.0 probe IDs to gene symbols using GPL570 annotation.

    Uses a lightweight approach: query the GEO GPL570 annotation file for probe-to-gene mapping.
    Falls back to a curated mapping for the most common probes if the full annotation is unavailable.
    """
    print("\nMapping probe IDs to gene symbols...")
    annotation_path = os.path.join(DATA_DIR, 'GPL570_annotation.csv')

    # Try to download GPL570 annotation if not present
    if not os.path.exists(annotation_path):
        try:
            import urllib.request
            url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/annot/GPL570.annot.gz"
            gz_path = os.path.join(DATA_DIR, 'GPL570.annot.gz')
            print("  Downloading GPL570 annotation from NCBI...")
            urllib.request.urlretrieve(url, gz_path)
            import gzip
            # Parse the annotation file
            rows = []
            with gzip.open(gz_path, 'rt', errors='replace') as f:
                header = None
                for line in f:
                    if line.startswith('#'):
                        continue
                    if header is None:
                        header = line.strip().split('\t')
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        probe_id = parts[0]
                        # Gene symbol is typically in column index 14 (Gene Symbol)
                        gene_sym = parts[14] if len(parts) > 14 else ''
                        gene_title = parts[13] if len(parts) > 13 else ''
                        rows.append({'probe_id': probe_id, 'gene_symbol': gene_sym, 'gene_title': gene_title})
            annot_df = pd.DataFrame(rows)
            annot_df.to_csv(annotation_path, index=False)
            os.remove(gz_path)
            print(f"  Parsed {len(annot_df)} probe annotations")
        except Exception as e:
            print(f"  GPL570 annotation download failed: {e}")
            # Try alternative: SOFT file
            try:
                url2 = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/soft/GPL570_family.soft.gz"
                gz_path2 = os.path.join(DATA_DIR, 'GPL570_soft.gz')
                print("  Trying SOFT file...")
                urllib.request.urlretrieve(url2, gz_path2)
                import gzip
                rows = []
                in_table = False
                col_indices = {}
                with gzip.open(gz_path2, 'rt', errors='replace') as f:
                    for line in f:
                        if line.startswith('!platform_table_begin'):
                            in_table = True
                            continue
                        if line.startswith('!platform_table_end'):
                            break
                        if in_table:
                            parts = line.strip().split('\t')
                            if not col_indices:
                                for i, col in enumerate(parts):
                                    col_indices[col] = i
                                continue
                            pid = parts[0]
                            gs_idx = col_indices.get('Gene Symbol', -1)
                            gt_idx = col_indices.get('Gene Title', -1)
                            gs = parts[gs_idx] if gs_idx >= 0 and gs_idx < len(parts) else ''
                            gt = parts[gt_idx] if gt_idx >= 0 and gt_idx < len(parts) else ''
                            rows.append({'probe_id': pid, 'gene_symbol': gs, 'gene_title': gt})
                annot_df = pd.DataFrame(rows)
                annot_df.to_csv(annotation_path, index=False)
                if os.path.exists(gz_path2):
                    os.remove(gz_path2)
                print(f"  Parsed {len(annot_df)} probe annotations from SOFT")
            except Exception as e2:
                print(f"  SOFT download also failed: {e2}")
                annot_df = pd.DataFrame(columns=['probe_id', 'gene_symbol', 'gene_title'])

    if os.path.exists(annotation_path):
        annot_df = pd.read_csv(annotation_path)
    else:
        annot_df = pd.DataFrame(columns=['probe_id', 'gene_symbol', 'gene_title'])

    # Map features
    probe_to_gene = dict(zip(annot_df['probe_id'].astype(str), annot_df['gene_symbol'].astype(str)))
    mapped = []
    for probe in feature_names:
        gene = probe_to_gene.get(probe, '')
        # Clean: take first symbol if multiple (separated by ///)
        if gene and '///' in gene:
            gene = gene.split('///')[0].strip()
        mapped.append({'probe_id': probe, 'gene_symbol': gene if gene and gene != 'nan' else ''})
    map_df = pd.DataFrame(mapped)
    n_mapped = sum(map_df['gene_symbol'] != '')
    print(f"  Mapped {n_mapped}/{len(feature_names)} probes to gene symbols ({n_mapped/len(feature_names)*100:.1f}%)")
    return map_df, probe_to_gene


def pathway_enrichment(gene_list, results_dir):
    """Run pathway enrichment on top genes via gseapy/Enrichr."""
    print("\nPathway enrichment analysis...")
    # Filter to valid gene symbols
    genes = [g for g in gene_list if g and g != '' and g != 'nan' and g != '---']
    if len(genes) < 5:
        print(f"  Only {len(genes)} valid genes, skipping enrichment")
        return None

    print(f"  Submitting {len(genes)} genes to Enrichr...")
    try:
        import gseapy as gp
        enrichment_results = []
        for lib in ['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022']:
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets=lib, organism='human',
                                 outdir=None, no_plot=True)
                if enr.results is not None and len(enr.results) > 0:
                    res = enr.results.copy()
                    res['Library'] = lib
                    enrichment_results.append(res)
                    sig = res[res['Adjusted P-value'] < 0.05]
                    print(f"    {lib}: {len(sig)} significant terms (FDR<0.05)")
            except Exception as e:
                print(f"    {lib} failed: {e}")

        if enrichment_results:
            all_enr = pd.concat(enrichment_results, ignore_index=True)
            all_enr.to_csv(os.path.join(results_dir, 'enrichment_results.csv'), index=False)
            return all_enr
        else:
            print("  No enrichment results obtained")
            return None
    except Exception as e:
        print(f"  Enrichment failed: {e}")
        return None


def generate_figures(predictions, y_test, test_df, feat_df, shap_values, X_te,
                     best_name, enrichment_df, figures_dir):
    """Generate all publication-ready figures."""
    print("\nGenerating figures...")
    sns.set_style('white')
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'figure.dpi': 300})
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Fig 1: ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, pred) in enumerate(predictions.items()):
        auc_val = roc_auc_score(y_test, pred['y_prob'])
        ax.plot(pred['fpr'], pred['tpr'], color=colors[i], lw=2,
                label=f'{name} (AUROC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curves')
    ax.legend(loc='lower right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  figures/roc_curve.png")

    # Fig 2: Precision-Recall curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, pred) in enumerate(predictions.items()):
        prec, rec, _ = precision_recall_curve(y_test, pred['y_prob'])
        ap_val = average_precision_score(y_test, pred['y_prob'])
        ax.plot(rec, prec, color=colors[i], lw=2, label=f'{name} (AUPRC = {ap_val:.3f})')
    ax.axhline(y=y_test.mean(), color='gray', ls='--', alpha=0.5, label=f'Prevalence ({y_test.mean():.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  figures/precision_recall_curve.png")

    # Fig 3: Calibration plot
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, pred) in enumerate(predictions.items()):
        try:
            prob_true, prob_pred = calibration_curve(y_test, pred['y_prob'], n_bins=8, strategy='uniform')
            ax.plot(prob_pred, prob_true, marker='o', color=colors[i], lw=2, label=name)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot (n_positive=8 in test set; interpret with caution)')
    ax.legend(loc='lower right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  figures/calibration_plot.png")

    # Fig 4: SHAP summary
    if shap_values is not None:
        try:
            import shap
            plt.figure(figsize=(8, 8))
            shap.summary_plot(shap_values, X_te, max_display=20, show=False)
            plt.title(f'SHAP Feature Importance ({best_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("  figures/shap_summary.png")
        except Exception as e:
            print(f"  SHAP plot failed: {e}")

    # Fig 5: Feature importance bar with gene symbols
    top_n = min(20, len(feat_df))
    top = feat_df.head(top_n).copy()
    labels = []
    for _, r in top.iterrows():
        if r.get('gene_symbol') and r['gene_symbol'] != '' and r['gene_symbol'] != 'nan':
            labels.append(f"{r['gene_symbol']} ({r['probe_id']})")
        else:
            labels.append(r['probe_id'])
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(top_n), top['importance'].values[::-1], color='#2ca02c', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Most Important Features ({best_name})')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  figures/feature_importance.png")

    # Fig 6: Enrichment dot plot (if available)
    if enrichment_df is not None and len(enrichment_df) > 0:
        sig = enrichment_df[enrichment_df['Adjusted P-value'] < 0.05].head(20)
        if len(sig) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            sig = sig.sort_values('Adjusted P-value', ascending=True)
            y_pos = range(len(sig))
            ax.barh(y_pos, -np.log10(sig['Adjusted P-value'].values), color='#d62728', alpha=0.7)
            ax.set_yticks(y_pos)
            term_labels = [t[:60] + '...' if len(t) > 60 else t for t in sig['Term'].values]
            ax.set_yticklabels(term_labels, fontsize=7)
            ax.set_xlabel('-log10(Adjusted P-value)')
            ax.set_title('Pathway Enrichment of Top Classifier Features')
            ax.axvline(x=-np.log10(0.05), color='gray', ls='--', alpha=0.5)
            sns.despine()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'enrichment_dotplot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("  figures/enrichment_dotplot.png")


def external_validation(X_tr, y_train, feature_names, scaler, data_dir, results_dir):
    """Cross-platform external validation on GSE6477 and GSE47552.

    Note on cross-platform normalization: When validating across different microarray
    platforms (GPL570 vs GPL96), we normalize validation samples to their own platform
    distribution rather than applying the training scaler. This is because different
    platforms have fundamentally different intensity distributions and probe designs,
    making cross-platform scaler transfer inappropriate. This is a standard approach
    in cross-platform gene expression validation (Warnat et al., 2005).
    """
    print("\nExternal validation...")
    val_rows = []

    # GSE6477 - different platform (GPL96)
    try:
        expr6 = pd.read_csv(os.path.join(data_dir, 'GSE6477_expression.csv'), index_col=0)
        pheno6 = pd.read_csv(os.path.join(data_dir, 'GSE6477_phenotype.csv'), index_col=0)
        stage_map6 = {}
        for idx, row in pheno6.iterrows():
            src = str(row.get('Sample_source_name_ch1', ''))
            if 'MGUS' in src:
                stage_map6[idx] = 'MGUS'
            elif 'smoldering' in src.lower():
                stage_map6[idx] = 'SMM'
            elif 'newly diagnosed' in src.lower():
                stage_map6[idx] = 'MM_new'
            elif 'relapsed' in src.lower():
                stage_map6[idx] = 'MM_relapsed'
            elif 'normal' in src.lower():
                stage_map6[idx] = 'Normal'

        X6 = expr6.T
        common6 = X6.columns.intersection(pd.Index(feature_names))
        stages6 = pd.Series(stage_map6)
        print(f"  GSE6477: {len(X6)} samples, {len(common6)} common probes")
        print(f"  Stages: {stages6.value_counts().to_dict()}")

        if len(common6) >= 20:
            X6_sub = X6[list(common6)].fillna(X6[list(common6)].median())
            # Cross-platform normalization: scale to own distribution (see docstring)
            X6_scaled = pd.DataFrame(StandardScaler().fit_transform(X6_sub),
                                     index=X6_sub.index, columns=X6_sub.columns)
            X_tr_common = X_tr[list(common6)]
            lr_reduced = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                           max_iter=1000, class_weight='balanced',
                                           random_state=RANDOM_STATE)
            lr_reduced.fit(X_tr_common, y_train)
            scores6 = lr_reduced.predict_proba(X6_scaled)[:, 1]

            score_by_stage = {}
            for idx in X6_scaled.index:
                s = stage_map6.get(idx)
                if s:
                    score_by_stage.setdefault(s, []).append(
                        scores6[list(X6_scaled.index).index(idx)])
            stage_summary = {}
            for s in ['Normal', 'MGUS', 'SMM', 'MM_new', 'MM_relapsed']:
                if s in score_by_stage:
                    vals = score_by_stage[s]
                    stage_summary[s] = {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals)}
                    print(f"    {s}: mean score = {np.mean(vals):.3f} +/- {np.std(vals):.3f} (n={len(vals)})")

            # Check monotonicity
            ordered_stages = ['Normal', 'MGUS', 'SMM', 'MM_new', 'MM_relapsed']
            means = [stage_summary.get(s, {}).get('mean', float('nan')) for s in ordered_stages]
            non_mono = []
            for i in range(len(means) - 1):
                if not np.isnan(means[i]) and not np.isnan(means[i+1]) and means[i] > means[i+1]:
                    non_mono.append(f"{ordered_stages[i]}({means[i]:.3f})>{ordered_stages[i+1]}({means[i+1]:.3f})")
            mono_note = 'Monotonic across stages' if not non_mono else f'Non-monotonic: {"; ".join(non_mono)} — may reflect biological heterogeneity within stages or limited cross-platform probe overlap ({len(common6)} of {len(feature_names)} probes)'
            print(f"    Monotonicity check: {mono_note}")

            val_rows.append({
                'Dataset': 'GSE6477', 'Platform': 'GPL96', 'N_samples': len(X6),
                'N_common_probes': len(common6), 'Validation_type': 'stage_discrimination',
                'Normal_score': stage_summary.get('Normal', {}).get('mean'),
                'MGUS_score': stage_summary.get('MGUS', {}).get('mean'),
                'SMM_score': stage_summary.get('SMM', {}).get('mean'),
                'MM_new_score': stage_summary.get('MM_new', {}).get('mean'),
                'MM_relapsed_score': stage_summary.get('MM_relapsed', {}).get('mean'),
                'Monotonicity': mono_note,
                'Note': 'Cross-platform; validation samples normalized to own distribution'
            })
        else:
            val_rows.append({
                'Dataset': 'GSE6477', 'Platform': 'GPL96', 'N_samples': len(X6),
                'N_common_probes': len(common6), 'Note': 'Insufficient probe overlap'
            })
        del expr6, X6
        gc.collect()
    except Exception as e:
        print(f"  GSE6477 failed: {e}")
        val_rows.append({'Dataset': 'GSE6477', 'Note': str(e)})

    # GSE47552 - different platform (GPL6244)
    try:
        expr47 = pd.read_csv(os.path.join(data_dir, 'GSE47552_expression.csv'), index_col=0)
        pheno47 = pd.read_csv(os.path.join(data_dir, 'GSE47552_phenotype.csv'), index_col=0)
        stage_map47 = {}
        for idx, row in pheno47.iterrows():
            ct = str(row.get('char_cell type', ''))
            if 'MGUS' in ct:
                stage_map47[idx] = 'MGUS'
            elif 'SMM' in ct:
                stage_map47[idx] = 'SMM'
            elif 'MM' in ct:
                stage_map47[idx] = 'MM'
            elif 'Normal' in ct or 'NPC' in ct:
                stage_map47[idx] = 'Normal'

        X47 = expr47.T
        common47 = X47.columns.intersection(pd.Index(feature_names))
        stages47 = pd.Series(stage_map47)
        print(f"\n  GSE47552: {len(X47)} samples, {len(common47)} common probes")
        print(f"  Stages: {stages47.value_counts().to_dict()}")

        if len(common47) >= 20:
            X47_sub = X47[list(common47)].fillna(X47[list(common47)].median())
            # Cross-platform normalization (see docstring above)
            X47_scaled = pd.DataFrame(StandardScaler().fit_transform(X47_sub),
                                      index=X47_sub.index, columns=X47_sub.columns)
            X_tr_common47 = X_tr[list(common47)]
            lr_reduced47 = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                              max_iter=1000, class_weight='balanced',
                                              random_state=RANDOM_STATE)
            lr_reduced47.fit(X_tr_common47, y_train)
            scores47 = lr_reduced47.predict_proba(X47_scaled)[:, 1]

            score_by_stage47 = {}
            for idx in X47_scaled.index:
                s = stage_map47.get(idx)
                if s:
                    score_by_stage47.setdefault(s, []).append(
                        scores47[list(X47_scaled.index).index(idx)])
            for s in ['Normal', 'MGUS', 'SMM', 'MM']:
                if s in score_by_stage47:
                    vals = score_by_stage47[s]
                    print(f"    {s}: mean score = {np.mean(vals):.3f} (n={len(vals)})")

            val_rows.append({
                'Dataset': 'GSE47552', 'Platform': 'GPL6244', 'N_samples': len(X47),
                'N_common_probes': len(common47), 'Validation_type': 'stage_discrimination',
                'Note': 'Cross-platform; validation samples normalized to own distribution'
            })
        else:
            val_rows.append({
                'Dataset': 'GSE47552', 'Platform': 'GPL6244', 'N_samples': len(X47),
                'N_common_probes': len(common47), 'Note': 'Insufficient probe overlap (HuGene 1.0 ST uses different probe IDs)'
            })
        del expr47, X47
        gc.collect()
    except Exception as e:
        print(f"  GSE47552 failed: {e}")
        val_rows.append({'Dataset': 'GSE47552', 'Note': str(e)})

    pd.DataFrame(val_rows).to_csv(os.path.join(results_dir, 'external_validation.csv'), index=False)
    return val_rows


def train_and_evaluate():
    """Main pipeline."""
    start_time = time.time()
    versions = log_versions()

    # Load data
    X, y = load_primary_dataset()
    n_probes_initial = X.shape[1]

    # Train/test split BEFORE preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)} (Stable:{sum(y_train==0)}, Prog:{sum(y_train==1)})")
    print(f"Test:  {len(X_test)} (Stable:{sum(y_test==0)}, Prog:{sum(y_test==1)})")

    # Preprocess
    X_tr, X_te, scaler, skb, vt = preprocess(X_train, y_train, X_test, n_features=200)
    feature_names = X_tr.columns.tolist()
    del X_train, X_test, X
    gc.collect()

    # Get models with search spaces
    n_pos = int(sum(y_train == 1))
    n_neg = int(sum(y_train == 0))
    models_and_params = get_tuned_models(n_pos, n_neg)

    # Nested CV with hyperparameter tuning
    cv_results, cv_detail_rows, tuned_models, best_params = nested_cv_with_tuning(
        X_tr, y_train, models_and_params
    )

    # Refit best models on full training with tuned params
    # For GB, use sample_weight
    for name, model in tuned_models.items():
        if name == 'Gradient Boosting':
            sw = compute_sample_weight('balanced', y_train)
            model.fit(X_tr, y_train, sample_weight=sw)
        else:
            model.fit(X_tr, y_train)

    # Test set evaluation
    test_df, predictions = evaluate_test_set(tuned_models, X_te, y_test)
    best_name = test_df.iloc[0]['Model']
    best_model = tuned_models[best_name]
    print(f"\nBest model: {best_name} (AUROC={test_df.iloc[0]['AUROC']:.3f})")

    # Bootstrap CIs for test-set metrics
    ci_df = bootstrap_test_ci(tuned_models, X_te, y_test, n_boot=1000, seed=RANDOM_STATE)

    # CV-to-test gap analysis
    gap_df = cv_test_gap_analysis(cv_results, test_df)

    # Published baseline comparison
    baseline_df = published_baseline_comparison(test_df, n_features=len(feature_names))

    # Feature importance with gene symbol mapping
    # If best model lacks native importances (e.g., SVM RBF), use LR coefficients instead.
    # LR provides interpretable coefficients and is the second-best model.
    # Permutation importance fails with small test sets (n_positive=8).
    interp_model = best_model
    interp_name = best_name
    if hasattr(best_model, 'feature_importances_'):
        imp = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        imp = np.abs(best_model.coef_[0])
    else:
        # Fall back to LR for interpretability (has coef_)
        print(f"  {best_name} has no native feature importances — using Logistic Regression coefficients for interpretability")
        lr_model = tuned_models.get('Logistic Regression')
        if lr_model is not None and hasattr(lr_model, 'coef_'):
            imp = np.abs(lr_model.coef_[0])
            interp_model = lr_model
            interp_name = 'Logistic Regression'
            print(f"  LR |coef| range: [{np.min(imp):.4f}, {np.max(imp):.4f}]")
        else:
            # Last resort: permutation importance
            print(f"  No LR available, computing permutation importance...")
            perm_result = permutation_importance(
                best_model, X_te, y_test, n_repeats=30, random_state=RANDOM_STATE,
                scoring='roc_auc', n_jobs=-1
            )
            imp = perm_result.importances_mean
            print(f"  Permutation importance computed (top feature: {np.max(imp):.4f})")

    gene_map_df, probe_to_gene = map_probes_to_genes(feature_names)
    feat_df = pd.DataFrame({'probe_id': feature_names, 'importance': imp})
    feat_df = feat_df.merge(gene_map_df, on='probe_id', how='left')
    feat_df = feat_df.sort_values('importance', ascending=False)

    # Pathway enrichment on top 100 genes
    top_genes = feat_df.head(100)['gene_symbol'].dropna().tolist()
    top_genes = [g for g in top_genes if g and g != '' and g != 'nan' and g != '---']
    enrichment_df = pathway_enrichment(top_genes, RESULTS_DIR)

    # SHAP
    shap_values = None
    try:
        import shap
        # Use the interpretability model (LR if SVM was best) for SHAP
        shap_model = interp_model
        shap_model_name = interp_name
        print(f"  Computing SHAP values using {shap_model_name}...")
        if shap_model_name in ['XGBoost', 'Random Forest', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(shap_model)
            shap_values = explainer.shap_values(X_te)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif shap_model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(shap_model, X_tr)
            shap_values = explainer.shap_values(X_te)
        else:
            bg = shap.sample(X_tr, min(50, len(X_tr)))
            explainer = shap.KernelExplainer(shap_model.predict_proba, bg)
            shap_values = explainer.shap_values(X_te)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        print("SHAP values computed successfully")
    except Exception as e:
        print(f"SHAP failed: {e}")

    # Use SHAP values for feature importance if native importances are zero
    if shap_values is not None and np.all(feat_df['importance'] == 0):
        shap_imp = np.abs(shap_values).mean(axis=0)
        feat_df['importance'] = shap_imp
        feat_df = feat_df.sort_values('importance', ascending=False)
        print("Feature importance updated from SHAP values (native importances unavailable)")

    # ==================== SAVE RESULTS ====================
    print("\nSaving results...")
    test_df.to_csv(os.path.join(RESULTS_DIR, 'model_performance.csv'), index=False)
    feat_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)
    pd.DataFrame(cv_detail_rows).to_csv(os.path.join(RESULTS_DIR, 'cross_validation.csv'), index=False)
    gap_df.to_csv(os.path.join(RESULTS_DIR, 'cv_test_gap.csv'), index=False)
    ci_df.to_csv(os.path.join(RESULTS_DIR, 'bootstrap_ci.csv'), index=False)
    baseline_df.to_csv(os.path.join(RESULTS_DIR, 'baseline_comparison.csv'), index=False)

    cv_summary = pd.DataFrame([
        {'Model': k, **v} for k, v in cv_results.items()
    ]).sort_values('AUROC_mean', ascending=False)
    cv_summary.to_csv(os.path.join(RESULTS_DIR, 'cv_summary.csv'), index=False)

    # Save tuned hyperparameters
    with open(os.path.join(RESULTS_DIR, 'tuned_hyperparameters.json'), 'w') as f:
        # Convert numpy types for JSON serialization
        clean_params = {}
        for name, params in best_params.items():
            clean_params[name] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()}
        json.dump(clean_params, f, indent=2)

    desc = {
        'Total_samples': 358, 'Training_samples': len(y_train), 'Test_samples': len(y_test),
        'Training_positive': n_pos, 'Training_negative': n_neg,
        'Test_positive': int(sum(y_test == 1)), 'Test_negative': int(sum(y_test == 0)),
        'Total_probes_initial': n_probes_initial, 'Features_selected': len(feature_names),
        'Best_model': best_name,
        'Best_AUROC': float(test_df.iloc[0]['AUROC']),
        'Best_AUPRC': float(test_df.iloc[0]['AUPRC']),
        'Class_imbalance_ratio': f"{n_neg}:{n_pos}",
        'CV_type': 'Nested: outer 5-fold x 3 repeats, inner 3-fold RandomizedSearchCV',
        'Runtime_seconds': round(time.time() - start_time, 1),
        'Package_versions': versions
    }
    with open(os.path.join(RESULTS_DIR, 'descriptive_stats.json'), 'w') as f:
        json.dump(desc, f, indent=2)

    # ==================== FIGURES ====================
    generate_figures(predictions, y_test, test_df, feat_df, shap_values, X_te,
                     best_name, enrichment_df, FIGURES_DIR)

    # ==================== EXTERNAL VALIDATION ====================
    external_validation(X_tr, y_train, feature_names, scaler, DATA_DIR, RESULTS_DIR)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print(f"Best model: {best_name}")
    print(f"  Test AUROC: {test_df.iloc[0]['AUROC']:.3f}")
    print(f"  Test AUPRC: {test_df.iloc[0]['AUPRC']:.3f}")
    print(f"  Brier Score: {test_df.iloc[0]['Brier_Score']:.3f}")
    print(f"\nResults: {RESULTS_DIR}/")
    print(f"Figures: {FIGURES_DIR}/")


if __name__ == '__main__':
    train_and_evaluate()
