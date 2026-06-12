"""
Advanced Feature Engineering - Plan 1: Domain-Specific Risk Scores
Produces feature set with ~40 features including:
- Base features (18)
- Ratio features (5 from baseline)
- Risk scores (7 domain-specific)
- Additional raw features (10)
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

# Ratio features (5 from baseline)
RATIO_FEATURES = [
    'loan_to_income',
    'payment_to_income',
    'credit_age_months',
    'revol_util_ratio',
    'total_credit_lines'
]

# Additional raw features (10 from baseline)
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

# Advanced: Domain-specific risk scores (7)
ADVANCED_FEATURES = [
    'delinquency_score',
    'credit_diversity',
    'emp_length_score',
    'recent_activity_score',
    'utilization_pressure',
    'credit_age_bucket',
    'income_verified'
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
    logger.info("Adding ratio features")
    
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
    """Add additional raw features"""
    logger.info("Adding additional raw features")
    present = [f for f in RAW_FEATURES if f in df.columns]
    missing = [f for f in RAW_FEATURES if f not in df.columns]
    logger.info(f"  Added {len(present)} raw features")
    if missing:
        logger.warning(f"  Missing: {missing}")
    return df

def add_advanced_features(df, logger):
    """Add domain-specific risk scores"""
    logger.info("Adding advanced risk score features")
    
    # 1. Credit health score (weighted delinquency)
    df['delinquency_score'] = (
        df['num_tl_120dpd_2m'].fillna(0) * 5 + 
        df['num_tl_90g_dpd_24m'].fillna(0) * 3 + 
        df['num_tl_30dpd'].fillna(0) * 1
    )
    
    # 2. Credit diversity score
    df['credit_diversity'] = (
        df['num_bc_tl'].fillna(0) + 
        df['num_il_tl'].fillna(0) + 
        df['num_rev_accts'].fillna(0) + 
        df['num_op_rev_tl'].fillna(0)
    )
    
    # 3. Income stability score (employment length)
    emp_map = {
        '<1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4,
        '4 years': 5, '5 years': 6, '6 years': 7, '7 years': 8,
        '8 years': 9, '9 years': 10, '10+ years': 11
    }
    df['emp_length_score'] = df['emp_length'].map(emp_map).fillna(0)
    
    # 4. Recent activity score (lower is better for credit risk)
    df['recent_activity_score'] = (
        df['mths_since_recent_inq'].fillna(0) + 
        df['mths_since_recent_bc'].fillna(0)
    )
    
    # 5. Utilization pressure score
    df['utilization_pressure'] = (
        (df['revol_util'].fillna(0) / 100) * 
        (df['revol_bal'].fillna(0) / df['total_rev_hi_lim'].clip(lower=1))
    )
    
    # 6. Credit age bucket (categorical)
    df['credit_age_bucket'] = pd.cut(
        df['credit_age_months'],
        bins=[0, 24, 60, 120, 999],
        labels=['0-2y', '2-5y', '5-10y', '10+y']
    )
    
    # 7. Income verified flag
    df['income_verified'] = (df['verification_status'] == 'Verified').astype(int)
    
    logger.info(f"  Added {len(ADVANCED_FEATURES)} advanced features")
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

def build_feature_set(df, logger):
    """Build advanced feature set"""
    
    # Start with base features
    continuous_cols = BASE_CONTINUOUS.copy()
    categorical_cols = BASE_CATEGORICAL.copy()
    
    # Add ratio features
    df = add_ratio_features(df, logger)
    continuous_cols.extend(RATIO_FEATURES)
    
    # Add raw features
    df = add_raw_features(df, logger)
    for f in RAW_FEATURES:
        if f in df.columns and f not in continuous_cols:
            continuous_cols.append(f)
    
    # Add advanced features
    df = add_advanced_features(df, logger)
    # Add advanced continuous features
    advanced_cont = ['delinquency_score', 'credit_diversity', 'emp_length_score', 
                     'recent_activity_score', 'utilization_pressure']
    continuous_cols.extend([c for c in advanced_cont if c in df.columns])
    # Add advanced categorical
    if 'credit_age_bucket' in df.columns:
        categorical_cols.append('credit_age_bucket')
    if 'income_verified' in df.columns:
        continuous_cols.append('income_verified')
    
    logger.info(f"Total features: {len(continuous_cols)} continuous + {len(categorical_cols)} categorical = {len(continuous_cols) + len(categorical_cols)} total features")
    
    # Impute derived features
    all_derived = [col for col in continuous_cols if col not in BASE_CONTINUOUS]
    for col in all_derived:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"  Imputed derived {col}: {df[col].isnull().sum()} missing with median={median_val:.2f}")
    
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
    
    # Metadata
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
    feature_df.to_parquet(output_path, compression='snappy', index=False)
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
    parser = argparse.ArgumentParser(description='Advanced feature engineering - Plan 1')
    parser.add_argument('--output', type=str, default='advanced_plan1.parquet',
                        help='Output filename')
    parser.add_argument('--no_clip', action='store_true',
                        help='Skip clipping extreme values')
    
    args = parser.parse_args()
    
    if not args.output.endswith('.parquet'):
        args.output += '.parquet'
    
    logger = setup_logging()
    logger.info("Starting Advanced Plan 1 feature engineering")
    
    df = load_data(logger)
    all_features = BASE_CONTINUOUS + BASE_CATEGORICAL + BASE_DATE
    df = handle_missing(df, all_features, logger)
    if not args.no_clip:
        df = clip_extremes(df, logger)
    
    feature_df, metadata, scaler, encoders = build_feature_set(df, logger)
    
    output_path = Path(PROCESSED_DIR) / args.output
    save_output(feature_df, metadata, output_path, logger)
    
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING COMPLETE (Advanced Plan 1)")
    logger.info(f"Features: {metadata['n_features']}")
    logger.info(f"Samples: {metadata['n_samples']:,}")
    logger.info(f"Default rate: {metadata['default_rate']:.2%}")
    logger.info("="*60)

if __name__ == "__main__":
    main()