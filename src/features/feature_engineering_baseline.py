"""
Baseline Feature Engineering - Plan 1 & 2
Produces two feature sets in Parquet format:
- Plan 1: 18 base + 5 ratio features = 23 features
- Plan 2: 18 base + 10 raw features = 28 features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import logging
from datetime import datetime

# =============================================
# Configuration
# =============================================

RAW_DATA_PATH = r"D:\contrastive-credit-representations\data\raw\accepted_2007_to_2018Q4.parquet"
PROCESSED_DIR = r"D:\contrastive-credit-representations\data\processed"
OUTPUT_NAME = "baseline_plan1.parquet"

# Base features (18 from your current set)
BASE_CONTINUOUS = [
    'fico_range_low', 'fico_range_high', 'dti', 'revol_util',
    'annual_inc', 'loan_amnt', 'inq_last_6mths', 'delinq_2yrs',
    'open_acc', 'total_acc', 'pub_rec', 'collections_12_mths_ex_med',
    'acc_now_delinq'
]

BASE_CATEGORICAL = [
    'home_ownership', 'purpose', 'term', 'verification_status'
]

BASE_DATE = ['earliest_cr_line']

# Plan 1: Ratio features (5 derived)
RATIO_FEATURES = [
    'loan_to_income',
    'payment_to_income',
    'credit_age_months',
    'revol_util_ratio',
    'total_credit_lines'
]

# Plan 2: Additional raw features (10)
RAW_FEATURES = [
    'tot_cur_bal',
    'total_bc_limit',
    'num_il_tl',
    'num_bc_tl',
    'bc_util',
    'num_rev_accts',
    'chargeoff_within_12_mths',
    'pub_rec_bankruptcies',
    'mort_acc',
    'mo_sin_old_rev_tl_op'
]

# =============================================
# Logging Setup
# =============================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(PROCESSED_DIR) / 'feature_engineering.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# =============================================
# Core Preprocessing
# =============================================

def load_data(logger):
    """Load raw parquet data"""
    logger.info(f"Loading data from {RAW_DATA_PATH}")
    df = pd.read_parquet(RAW_DATA_PATH, engine='pyarrow')
    logger.info(f"Loaded {len(df):,} rows")
    return df

def handle_missing(df, cols, logger):
    """Median imputation for continuous, mode for categorical"""
    for col in cols:
        if col not in df.columns:
            continue
        missing = df[col].isnull().sum()
        if missing == 0:
            continue
        
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"  Imputed {col}: {missing} missing with median={median_val:.2f}")
        else:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug(f"  Imputed {col}: {missing} missing with mode='{mode_val}'")
    return df

def clip_extremes(df, logger):
    """Clip extreme values to prevent outliers"""
    clips = [
        ('dti', 60),
        ('revol_util', 120),
        ('inq_last_6mths', 10),
        ('annual_inc', 0.99),
        ('loan_amnt', 0.99)
    ]
    
    for col, threshold in clips:
        if col not in df.columns:
            continue
        if isinstance(threshold, float):
            cap = df[col].quantile(threshold)
        else:
            cap = threshold
        df[col] = df[col].clip(upper=cap)
        logger.debug(f"  Clipped {col} at {cap:.2f}")
    return df

def create_target(df, logger):
    """Create binary target from loan_status"""
    default_statuses = [
        'Charged Off', 'Default', 'Late (31-120 days)',
        'Does not meet the credit policy. Status:Charged Off'
    ]
    y = df['loan_status'].isin(default_statuses).astype(int)
    logger.info(f"Default rate: {y.mean():.2%} ({y.sum():,} defaults)")
    return y

# =============================================
# Feature Engineering Functions
# =============================================

def add_ratio_features(df, logger):
    """Add Plan 1 ratio features"""
    logger.info("Adding ratio features (Plan 1)")
    
    # Loan-to-income
    df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].clip(lower=1)
    
    # Payment-to-income
    df['payment_to_income'] = df['installment'] / (df['annual_inc'] / 12).clip(lower=1)
    
    # Credit age in months
    ref_date = pd.Timestamp('2018-12-31')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
    median_date = df['earliest_cr_line'].dropna().median()
    df['earliest_cr_line'] = df['earliest_cr_line'].fillna(median_date)
    df['credit_age_months'] = ((ref_date - df['earliest_cr_line']).dt.days / 30.44).round()
    
    # Revolving utilization ratio (cleaner)
    df['revol_util_ratio'] = df['revol_bal'] / df['total_rev_hi_lim'].clip(lower=1)
    
    # Total credit lines
    df['total_credit_lines'] = df['open_acc'] + df['total_acc']
    
    logger.info(f"  Added {len(RATIO_FEATURES)} ratio features")
    return df

def add_raw_features(df, logger):
    """Add Plan 2 additional raw features"""
    logger.info("Adding additional raw features (Plan 2)")
    
    # Features already in df, just log which ones exist
    present = [f for f in RAW_FEATURES if f in df.columns]
    missing = [f for f in RAW_FEATURES if f not in df.columns]
    
    logger.info(f"  Added {len(present)} raw features")
    if missing:
        logger.warning(f"  Missing: {missing}")
    
    return df

def encode_categorical(df, cols, logger):
    """Label encode categorical features"""
    encoders = {}
    for col in cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.debug(f"  Encoded {col}: {len(le.classes_)} unique values")
    return df, encoders

def standardize_continuous(df, cols, logger):
    """Standardize continuous features"""
    scaler = StandardScaler()
    X_cont = scaler.fit_transform(df[cols])
    logger.debug(f"  Standardized {len(cols)} features")
    return X_cont, scaler

# =============================================
# Main Feature Engineering Pipeline
# =============================================
def build_feature_set(df, plan, logger):
    """Build feature set for Plan 1 or Plan 2"""
    
    # Start with base features
    continuous_cols = BASE_CONTINUOUS.copy()
    categorical_cols = BASE_CATEGORICAL.copy()
    
    # Add derived features based on plan
    if plan == 1:
        df = add_ratio_features(df, logger)
        continuous_cols.extend(RATIO_FEATURES)
        logger.info(f"Plan 1: {len(continuous_cols)} continuous + {len(categorical_cols)} categorical = {len(continuous_cols) + len(categorical_cols)} total features")
    
    elif plan == 2:
        df = add_raw_features(df, logger)
        # Add raw features that exist
        for f in RAW_FEATURES:
            if f in df.columns and f not in continuous_cols:
                continuous_cols.append(f)
        logger.info(f"Plan 2: {len(continuous_cols)} continuous + {len(categorical_cols)} categorical = {len(continuous_cols) + len(categorical_cols)} total features")
    
    # ===== FIX: IMPUTE DERIVED FEATURES =====
    # Identify derived features (all continuous except BASE_CONTINUOUS)
    derived_cols = [col for col in continuous_cols if col not in BASE_CONTINUOUS]
    
    for col in derived_cols:
        if col not in df.columns:
            continue
        missing = df[col].isnull().sum()
        if missing > 0:
            # For ratio features, use median imputation
            if col in RATIO_FEATURES:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"  Imputed derived {col}: {missing} missing with median={median_val:.2f}")
            # For raw features, use median
            elif col in RAW_FEATURES:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"  Imputed raw {col}: {missing} missing with median={median_val:.2f}")
            # For any other derived, use median
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"  Imputed {col}: {missing} missing with median={median_val:.2f}")
    # =====================================
    
    # Encode categoricals
    df, encoders = encode_categorical(df, categorical_cols, logger)
    
    # Standardize continuous
    X_cont, scaler = standardize_continuous(df, continuous_cols, logger)
    
    # Get categorical matrix
    X_cat = df[categorical_cols].values
    
    # Get target
    y = create_target(df, logger)
    
    # Build final DataFrame for Parquet
    feature_df = pd.DataFrame(
        np.hstack([X_cont, X_cat]),
        columns=continuous_cols + categorical_cols
    )
    feature_df['target'] = y
    
    # Store metadata separately (optional, as JSON)
    metadata = {
        'continuous_cols': continuous_cols,
        'categorical_cols': categorical_cols,
        'n_features': len(continuous_cols) + len(categorical_cols),
        'n_samples': len(y),
        'default_rate': y.mean()
    }
    
    return feature_df, metadata, scaler, encoders

def save_output(feature_df, metadata, output_path, logger):
    """Save processed data to Parquet file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as Parquet with Snappy compression
    feature_df.to_parquet(output_path, compression='snappy', index=False)
    
    # Save metadata as JSON
    meta_path = output_path.with_suffix('.meta.json')
    import json
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Metadata saved to: {meta_path}")

# =============================================
# CLI Entry Point
# =============================================
def main():
    parser = argparse.ArgumentParser(description='Baseline feature engineering')
    parser.add_argument('--plan', type=int, choices=[1, 2], required=True,
                        help='1 = ratio features, 2 = additional raw features')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (e.g., baseline_plan1.parquet). If not provided, auto-generated as baseline_plan{plan}.parquet')
    parser.add_argument('--no_clip', action='store_true',
                        help='Skip clipping extreme values')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output is None:
        args.output = f"baseline_plan{args.plan}.parquet"
    
    # Auto-add .parquet extension if missing
    if not args.output.endswith('.parquet'):
        args.output += '.parquet'
    
    # Prevent accidental overwriting of existing files
    output_path = Path(PROCESSED_DIR) / args.output
    if output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = output_path.stem + f"_backup_{timestamp}" + output_path.suffix
        backup_path = output_path.with_name(backup_name)
        output_path.rename(backup_path)
        print(f"⚠️ Existing file backed up to: {backup_path}")
    
    logger = setup_logging()
    logger.info(f"Starting Baseline Plan {args.plan} feature engineering")
    
    # Load data
    df = load_data(logger)
    
    # Preprocess
    all_features = BASE_CONTINUOUS + BASE_CATEGORICAL + BASE_DATE
    df = handle_missing(df, all_features, logger)
    if not args.no_clip:
        df = clip_extremes(df, logger)
    
    # Build feature set
    feature_df, metadata, scaler, encoders = build_feature_set(df, args.plan, logger)
    
    # ===== FIX: Check for NaN values =====
    nan_count = feature_df.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠️ Found {nan_count} NaN values in feature_df! Filling with 0...")
        feature_df = feature_df.fillna(0)
    # =====================================
    
    # Save
    save_output(feature_df, metadata, output_path, logger)
    
    logger.info("="*60)
    logger.info(f"FEATURE ENGINEERING COMPLETE (Plan {args.plan})")
    logger.info(f"Features: {metadata['n_features']}")
    logger.info(f"Samples: {metadata['n_samples']:,}")
    logger.info(f"Default rate: {metadata['default_rate']:.2%}")
    logger.info("="*60)



if __name__ == "__main__":
    main()