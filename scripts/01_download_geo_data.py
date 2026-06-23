#!/usr/bin/env python3
"""Download and parse GEO series matrix files for MGUS progression classifier.

Downloads preprocessed expression data from NCBI GEO for:
- GSE235356: Primary dataset (358 MGUS samples, GPL570)
- GSE6477: Validation (162 samples, multi-stage, GPL96)
- GSE47552: Validation (99 samples, multi-stage, GPL6244)
- GSE24080: Survival validation (559 MM samples, GPL570)

Usage:
    python3 01_download_geo_data.py
"""

import os
import gzip
import urllib.request
import pandas as pd
import numpy as np
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = {
    'GSE235356': {'platform': 'GPL570', 'desc': 'Primary MGUS (358 samples)'},
    'GSE6477': {'platform': 'GPL96', 'desc': 'Validation multi-stage (162 samples)'},
    'GSE47552': {'platform': 'GPL6244', 'desc': 'Validation multi-stage (99 samples)'},
    'GSE24080': {'platform': 'GPL570', 'desc': 'Survival validation MM (559 samples)'},
}


def download_series_matrix(gse_id):
    """Download series matrix file from GEO FTP."""
    gz_path = os.path.join(DATA_DIR, f'{gse_id}_series_matrix.txt.gz')
    if os.path.exists(gz_path):
        print(f"  {gz_path} already exists, skipping download")
        return gz_path

    # Construct FTP URL
    prefix = gse_id[:len(gse_id)-3] + 'nnn'
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, gz_path)
    print(f"  Saved to {gz_path} ({os.path.getsize(gz_path)/1024/1024:.1f} MB)")
    return gz_path


def parse_series_matrix(gz_path, gse_id):
    """Parse series matrix file into expression and phenotype DataFrames."""
    expr_path = os.path.join(DATA_DIR, f'{gse_id}_expression.csv')
    pheno_path = os.path.join(DATA_DIR, f'{gse_id}_phenotype.csv')

    if os.path.exists(expr_path) and os.path.exists(pheno_path):
        expr = pd.read_csv(expr_path, index_col=0)
        pheno = pd.read_csv(pheno_path, index_col=0)
        print(f"  Loaded existing: {expr.shape[0]} probes x {expr.shape[1]} samples")
        return expr, pheno

    print(f"  Parsing {gz_path}...")
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

    # Build expression matrix
    if expression_lines:
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
    else:
        print(f"  WARNING: No expression data found in {gz_path}")
        return None, None

    # Build phenotype table
    pheno_dict = {}
    for key, value_lists in metadata.items():
        if len(value_lists) == 1:
            pheno_dict[key] = value_lists[0]
        else:
            for i, vl in enumerate(value_lists):
                pheno_dict[f'{key}_{i}'] = vl

    # Find sample IDs in metadata
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

    # Save
    expr.to_csv(expr_path)
    pheno.to_csv(pheno_path)
    print(f"  Saved: {expr.shape[0]} probes x {expr.shape[1]} samples")
    return expr, pheno


def main():
    print("=" * 60)
    print("GEO Data Download and Parsing")
    print("=" * 60)

    for gse_id, info in DATASETS.items():
        print(f"\n{gse_id} ({info['desc']}):")
        try:
            gz_path = download_series_matrix(gse_id)
            expr, pheno = parse_series_matrix(gz_path, gse_id)
            if expr is not None:
                print(f"  Expression: {expr.shape}")
                print(f"  Phenotype: {pheno.shape}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Download complete. Data files in:", DATA_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
