"""
Baseline Feature Selection - Version 2
18 carefully selected features with full preprocessing pipeline
Based on rigorous EDA and credit risk literature
"""

import pandas as pd
import numpy as np
import os  # ← MISSING IMPORT
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================

RAW_DATA_PATH = r"D:\contrastive-credit-representations\data\raw\accepted_2007_to_2018Q4.parquet"
PROCESSED_DIR = r"D:\contrastive-credit-representations\data\processed"

# =============================================
# Feature Definition (Exact 18 features from your plan)
# =============================================

BASELINE_FEATURES = {
    'continuous': [
        'fico_range_low',      # Lower bound of FICO score
        'fico_range_high',     # Upper bound of FICO score
        'dti',                 # Debt-to-income ratio
        'revol_util',          # Revolving credit utilization
        'annual_inc',          # Self-reported annual income
        'loan_amnt',           # Requested loan amount
        'inq_last_6mths',      # Credit inquiries in last 6 months
        'delinq_2yrs',         # Delinquency incidents in past 2 years
        'open_acc',            # Number of open credit lines
        'total_acc',           # Total credit lines ever opened
        'pub_rec',             # Number of public records
        'collections_12_mths_ex_med',  # Collections excluding medical
        'acc_now_delinq',      # Accounts currently delinquent
    ],
    'categorical': [
        'home_ownership',      # RENT, MORTGAGE, OWN, OTHER
        'purpose',             # debt_consolidation, credit_card, small_business, etc.
        'term',                # 36 months, 60 months
        'verification_status'  # Not Verified, Source Verified, Verified
    ],
    'date_derived': [
        'earliest_cr_line'     # Will be converted to months since
    ]
}

# =============================================
# Preprocessing Functions
# =============================================

def preprocess_baseline_features(df=None, output_path=None):
    """
    Preprocess the 18 baseline features for contrastive learning.
    
    Args:
        df: Optional pre-loaded dataframe
        output_path: If provided, save processed data
    
    Returns:
        X_continuous: (n_samples, 13) standardized continuous features
        X_categorical: (n_samples, 4) label-encoded categorical features
        X_date: (n_samples, 1) date-derived feature
        y: (n_samples,) binary target (default=1, non-default=0)
        metadata: Dictionary with preprocessing objects
    """
    
    if df is None:
        print("Loading data...")
        df = pd.read_parquet(RAW_DATA_PATH, engine='pyarrow')
    
    print(f"Loaded {len(df):,} rows")
    
    # Create copy to avoid warnings
    data = df.copy()
    
    # ---- SELECT FEATURES ----
    continuous_cols = BASELINE_FEATURES['continuous']
    categorical_cols = BASELINE_FEATURES['categorical']
    date_cols = BASELINE_FEATURES['date_derived']
    
    print(f"\nFeature selection:")
    print(f"  Continuous: {len(continuous_cols)} features")
    print(f"  Categorical: {len(categorical_cols)} features")
    print(f"  Date-derived: {len(date_cols)} features")
    print(f"  Total: {len(continuous_cols) + len(categorical_cols) + len(date_cols)} features")
    
    # ---- HANDLE MISSING VALUES ----
    print("\nHandling missing values...")
    
    # Check missing rates first
    all_features = continuous_cols + categorical_cols + date_cols
    missing_rates = data[all_features].isnull().mean() * 100
    high_missing = missing_rates[missing_rates > 0]
    if len(high_missing) > 0:
        print("  Features with missing values:")
        for col, rate in high_missing.items():
            print(f"    {col}: {rate:.2f}% missing")
    
    # Continuous: median imputation
    for col in continuous_cols:
        if col in data.columns and data[col].isnull().sum() > 0:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            print(f"  Imputed {col}: {data[col].isnull().sum()} missing with median={median_val:.2f}")
    
    # Categorical: mode imputation
    for col in categorical_cols:
        if col in data.columns and data[col].isnull().sum() > 0:
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
            print(f"  Imputed {col}: filled with mode='{mode_val}'")
    
    # Date: fill with median date
    if 'earliest_cr_line' in data.columns:
        # Convert to datetime first
        data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'], errors='coerce')
        # Fill NaT with median date
        median_date = data['earliest_cr_line'].dropna().median()
        data['earliest_cr_line'] = data['earliest_cr_line'].fillna(median_date)
        print(f"  Imputed earliest_cr_line: filled with median date")
    
    # ---- CLIP EXTREME VALUES ----
    print("\nClipping extreme values...")
    
    # DTI: cap at 60%
    if 'dti' in data.columns:
        data['dti'] = data['dti'].clip(upper=60)
        print(f"  dti: capped at 60")
    
    # Revol_util: cap at 120%
    if 'revol_util' in data.columns:
        data['revol_util'] = data['revol_util'].clip(upper=120)
        print(f"  revol_util: capped at 120")
    
    # Inq_last_6mths: cap at 10
    if 'inq_last_6mths' in data.columns:
        data['inq_last_6mths'] = data['inq_last_6mths'].clip(upper=10)
        print(f"  inq_last_6mths: capped at 10")
    
    # Annual income: cap at 99th percentile
    if 'annual_inc' in data.columns:
        income_cap = data['annual_inc'].quantile(0.99)
        data['annual_inc'] = data['annual_inc'].clip(upper=income_cap)
        print(f"  annual_inc: capped at 99th percentile ({income_cap:.2f})")
    
    # Loan amount: cap at 99th percentile
    if 'loan_amnt' in data.columns:
        loan_cap = data['loan_amnt'].quantile(0.99)
        data['loan_amnt'] = data['loan_amnt'].clip(upper=loan_cap)
        print(f"  loan_amnt: capped at 99th percentile ({loan_cap:.2f})")
    
    # ---- DERIVE DATE FEATURES ----
    print("\nDeriving date features...")
    if 'earliest_cr_line' in data.columns:
        # Convert to months since earliest credit line
        reference_date = pd.Timestamp('2018-12-31')  # End of dataset
        data['months_since_earliest_cr_line'] = (
            (reference_date - data['earliest_cr_line']).dt.days / 30.44
        ).round().astype(int)
        # Add to continuous features
        continuous_cols.append('months_since_earliest_cr_line')
        print(f"  Derived months_since_earliest_cr_line from earliest_cr_line")
    
    # ---- ENCODE CATEGORICAL FEATURES ----
    print("\nEncoding categorical features...")
    encoders = {}
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
            print(f"  Encoded {col}: {len(le.classes_)} unique values")
    
    # ---- STANDARDIZE CONTINUOUS FEATURES ----
    print("\nStandardizing continuous features...")
    scaler = StandardScaler()
    X_continuous = scaler.fit_transform(data[continuous_cols])
    X_continuous = pd.DataFrame(X_continuous, columns=continuous_cols, index=data.index)
    
    # ---- COMBINE CATEGORICAL FEATURES ----
    X_categorical = data[categorical_cols]
    
    # ---- CREATE TARGET VARIABLE ----
    print("\nCreating target variable...")
    default_statuses = [
        'Charged Off', 
        'Default', 
        'Late (31-120 days)', 
        'Does not meet the credit policy. Status:Charged Off'
    ]
    data['default'] = data['loan_status'].isin(default_statuses).astype(int)
    y = data['default']
    
    print(f"  Default rate: {y.mean():.2%}")
    print(f"  Defaults: {y.sum():,}")
    print(f"  Non-defaults: {(1 - y).sum():,}")
    
    # ---- STORE METADATA ----
    metadata = {
        'continuous_cols': continuous_cols,
        'categorical_cols': categorical_cols,
        'n_continuous': len(continuous_cols),
        'n_categorical': len(categorical_cols),
        'total_features': len(continuous_cols) + len(categorical_cols),
        'scaler': scaler,
        'encoders': encoders,
        'default_statuses': default_statuses,
        'n_samples': len(data)
    }
    
    # ---- SAVE PROCESSED DATA ----
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays for faster loading
        np.savez_compressed(
            output_path,
            X_continuous=X_continuous.values,
            X_categorical=X_categorical.values,
            y=y.values,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols
        )
        print(f"\nSaved processed data to: {output_path}")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"  Samples: {len(data):,}")
    print(f"  Continuous features: {len(continuous_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"  Total features: {len(continuous_cols) + len(categorical_cols)}")
    print(f"  Default rate: {y.mean():.2%}")
    
    return X_continuous, X_categorical, y, metadata

# =============================================
# Feature Validation
# =============================================

def validate_features(X_continuous, X_categorical, y):
    """
    Print summary statistics for baseline features.
    """
    print("\n" + "="*60)
    print("BASELINE FEATURE VALIDATION")
    print("="*60)
    
    # Continuous features
    print("\nContinuous Features:")
    print(f"{'Feature':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for col in X_continuous.columns:
        print(f"{col:<30} {X_continuous[col].mean():>8.3f} "
              f"{X_continuous[col].std():>8.3f} "
              f"{X_continuous[col].min():>8.3f} "
              f"{X_continuous[col].max():>8.3f}")
    
    # Categorical features
    print("\nCategorical Features:")
    for col in X_categorical.columns:
        n_unique = X_categorical[col].nunique()
        print(f"\n  {col}: {n_unique} unique values")
        # Show top 3 categories
        top_cats = X_categorical[col].value_counts().head(3)
        for val, count in top_cats.items():
            print(f"    {val}: {count} ({count/len(X_categorical):.1%})")
    
    # Target distribution
    print(f"\nTarget Distribution:")
    print(f"  Non-default: {(y==0).sum():,} ({(y==0).mean():.1%})")
    print(f"  Default: {(y==1).sum():,} ({(y==1).mean():.1%})")
    
    # Missing values check
    total_missing = X_continuous.isnull().sum().sum() + X_categorical.isnull().sum().sum()
    print(f"\nTotal Missing Values: {total_missing}")
    if total_missing > 0:
        print("  ⚠️ WARNING: Missing values detected!")
        missing_cols = X_continuous.columns[X_continuous.isnull().any()].tolist()
        missing_cols += X_categorical.columns[X_categorical.isnull().any()].tolist()
        print(f"  Columns with missing values: {missing_cols}")
    else:
        print("  ✅ OK: No missing values")
    
    print("="*60)
    return

# =============================================
# Quick Start
# =============================================

if __name__ == "__main__":
    output_path = os.path.join(PROCESSED_DIR, "baseline_features_v2.npz")
    
    # Load and preprocess
    X_cont, X_cat, y, metadata = preprocess_baseline_features(output_path=output_path)
    
    # Validate
    validate_features(X_cont, X_cat, y)
    
    print(f"\n✅ Baseline features v2 ready for SSL experiments!")
    print(f"  Processed data saved to: {output_path}")