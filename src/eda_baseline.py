"""
Baseline Exploratory Data Analysis for Lending Club Dataset
- Analyzes all features in raw data
- Provides statistical properties
- Identifies missing values, cardinality, and data types
- Recommends features to keep or remove
- Includes visualizations for better understanding
- Optimized for Intel Core i5 with chunked processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================

ACCEPTED_PARQUET = r"D:\contrastive-credit-representations\data\raw\accepted_2007_to_2018Q4.parquet"
OUTPUT_DIR = r"D:\contrastive-credit-representations\eda_output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================
# Load Data
# =============================================

print("="*80)
print("LENDING CLUB BASELINE EDA")
print("="*80)

print("\nLoading accepted loans data...")
start_time = time.time()

# Load with pyarrow (handles categorical columns correctly)
df = pd.read_parquet(ACCEPTED_PARQUET, engine='pyarrow')

print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Load time: {time.time() - start_time:.2f} seconds")

# =============================================
# 1. Basic Dataset Information
# =============================================

print("\n" + "="*80)
print("1. DATASET OVERVIEW")
print("="*80)

print(f"\nShape: {df.shape}")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# =============================================
# 2. Column Types Analysis
# =============================================

print("\n" + "="*80)
print("2. COLUMN TYPES ANALYSIS")
print("="*80)

# Count by data type
dtype_counts = df.dtypes.value_counts()
print("\nData type distribution:")
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# Separate by type
numeric_cols = df.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
object_cols = df.select_dtypes(include=['object']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Object columns: {len(object_cols)}")
print(f"Datetime columns: {len(datetime_cols)}")

# Visualization: Column types distribution
fig, ax = plt.subplots(figsize=(10, 6))
dtype_counts.plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Column Types Distribution', fontsize=16)
ax.set_xlabel('Data Type')
ax.set_ylabel('Number of Columns')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(dtype_counts):
    ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'column_types_distribution.png'), dpi=150)
plt.close()
print(f"✓ Saved: column_types_distribution.png")

# =============================================
# 3. Missing Values Analysis
# =============================================

print("\n" + "="*80)
print("3. MISSING VALUES ANALYSIS")
print("="*80)

missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isna().sum().values,
    'Missing %': (df.isna().sum() / len(df) * 100).values,
    'Dtype': df.dtypes.values
})
missing_df = missing_df.sort_values('Missing %', ascending=False)

# Show top 20 columns with highest missing values
print("\nTop 20 columns with highest missing values:")
print(missing_df.head(20).to_string(index=False))

# Columns with >50% missing values
high_missing = missing_df[missing_df['Missing %'] > 50]
print(f"\nColumns with >50% missing values: {len(high_missing)}")
if len(high_missing) > 0:
    print(high_missing[['Column', 'Missing %']].to_string(index=False))

# Columns with 0 missing values
complete_cols = missing_df[missing_df['Missing %'] == 0]
print(f"\nColumns with 0 missing values: {len(complete_cols)}")

# Visualization: Missing values heatmap (top 50 columns)
fig, ax = plt.subplots(figsize=(12, 10))
top_missing = missing_df.head(50)
missing_matrix = df[top_missing['Column']].isna()
sns.heatmap(missing_matrix.T, cbar=True, ax=ax, cmap='viridis')
ax.set_title('Missing Values Heatmap (Top 50 Columns)', fontsize=16)
ax.set_xlabel('Rows (sample)')
ax.set_ylabel('Columns')
ax.tick_params(axis='x', labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values_heatmap.png'), dpi=150)
plt.close()
print(f"✓ Saved: missing_values_heatmap.png")

# Visualization: Missing percentage bar chart
fig, ax = plt.subplots(figsize=(12, 8))
top_30_missing = missing_df.head(30)
ax.barh(range(len(top_30_missing)), top_30_missing['Missing %'], color='coral')
ax.set_yticks(range(len(top_30_missing)))
ax.set_yticklabels(top_30_missing['Column'], fontsize=8)
ax.set_xlabel('Missing Percentage (%)')
ax.set_title('Top 30 Columns by Missing Percentage', fontsize=16)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'missing_percentage_bar.png'), dpi=150)
plt.close()
print(f"✓ Saved: missing_percentage_bar.png")

# =============================================
# 4. Cardinality Analysis (Unique Values)
# =============================================

print("\n" + "="*80)
print("4. CARDINALITY ANALYSIS (UNIQUE VALUES)")
print("="*80)

cardinality = pd.DataFrame({
    'Column': df.columns,
    'Unique Values': df.nunique().values,
    'Dtype': df.dtypes.values
})
cardinality = cardinality.sort_values('Unique Values', ascending=False)

# Show high cardinality columns
high_cardinality = cardinality[cardinality['Unique Values'] > 1000]
print(f"\nHigh cardinality columns (>1000 unique values): {len(high_cardinality)}")
print(high_cardinality[['Column', 'Unique Values', 'Dtype']].head(20).to_string(index=False))

# Show low cardinality columns
low_cardinality = cardinality[cardinality['Unique Values'] <= 10]
print(f"\nLow cardinality columns (≤10 unique values): {len(low_cardinality)}")
print(low_cardinality[['Column', 'Unique Values', 'Dtype']].to_string(index=False))

# Visualization: Cardinality distribution
fig, ax = plt.subplots(figsize=(12, 8))
unique_counts = cardinality['Unique Values']
unique_counts = unique_counts[unique_counts < 10000]  # Filter outliers for better visualization
ax.hist(unique_counts, bins=50, color='teal', edgecolor='black')
ax.set_title('Cardinality Distribution (Unique Values per Column)', fontsize=16)
ax.set_xlabel('Number of Unique Values')
ax.set_ylabel('Number of Columns')
ax.axvline(x=1000, color='red', linestyle='--', label='High Cardinality Threshold (1000)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cardinality_distribution.png'), dpi=150)
plt.close()
print(f"✓ Saved: cardinality_distribution.png")

# =============================================
# 5. Basic Statistics for Numeric Columns
# =============================================

print("\n" + "="*80)
print("5. NUMERIC COLUMNS STATISTICS")
print("="*80)

# Get numeric columns statistics
numeric_stats = df[numeric_cols].describe().T
numeric_stats = numeric_stats.sort_values('count', ascending=False)

print("\nFirst 20 numeric columns (sorted by non-null count):")
print(numeric_stats.head(20).to_string())

# Check for columns with all same value
constant_cols = []
for col in numeric_cols:
    if df[col].nunique() == 1:
        constant_cols.append(col)
print(f"\nConstant numeric columns (all same value): {len(constant_cols)}")
if constant_cols:
    print(f"  {constant_cols}")

# Visualization: Correlation matrix (top numeric columns)
if len(numeric_cols) > 0:
    # Select top numeric columns by variance (to avoid memory issues)
    top_numeric = numeric_cols[:20] if len(numeric_cols) > 20 else numeric_cols
    numeric_df = df[top_numeric].select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(14, 12))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True)
        ax.set_title('Correlation Matrix (Top Numeric Columns)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=150)
        plt.close()
        print(f"✓ Saved: correlation_matrix.png")

# =============================================
# 6. Target Variable Analysis (Default Rate)
# =============================================

print("\n" + "="*80)
print("6. TARGET VARIABLE ANALYSIS (DEFAULT RATE)")
print("="*80)

# Check loan_status column
if 'loan_status' in df.columns:
    status_counts = df['loan_status'].value_counts()
    print("\nLoan status distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Define default statuses
    default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']
    default_mask = df['loan_status'].isin(default_statuses)
    
    print(f"\nDefault rate (based on loan_status):")
    print(f"  Defaults: {default_mask.sum():,} ({default_mask.sum()/len(df)*100:.2f}%)")
    print(f"  Non-defaults: {(~default_mask).sum():,} ({(~default_mask).sum()/len(df)*100:.2f}%)")
    
    # Visualization: Loan status distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    status_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Loan Status Distribution', fontsize=16)
    ax.set_xlabel('Loan Status')
    ax.set_ylabel('Number of Loans')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(status_counts):
        ax.text(i, v + 1000, f'{v:,}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loan_status_distribution.png'), dpi=150)
    plt.close()
    print(f"✓ Saved: loan_status_distribution.png")
    
    # Visualization: Default vs Non-Default pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    default_counts = [default_mask.sum(), (~default_mask).sum()]
    labels = ['Default', 'Non-Default']
    colors = ['#ff6b6b', '#51cf66']
    ax.pie(default_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Default vs Non-Default Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'default_distribution.png'), dpi=150)
    plt.close()
    print(f"✓ Saved: default_distribution.png")

# =============================================
# 7. Key Numeric Features Distribution
# =============================================

print("\n" + "="*80)
print("7. KEY NUMERIC FEATURES DISTRIBUTION")
print("="*80)

# Select key numeric features for visualization
key_numeric_features = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'fico_range_low']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(key_numeric_features):
    if col in df.columns:
        df[col].dropna().hist(ax=axes[i], bins=50, color='purple', edgecolor='black')
        axes[i].set_title(f'{col} Distribution', fontsize=14)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        axes[i].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
        axes[i].legend()

# Hide empty subplot
if len(key_numeric_features) < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'key_numeric_distributions.png'), dpi=150)
plt.close()
print(f"✓ Saved: key_numeric_distributions.png")

# =============================================
# 8. Feature Recommendations
# =============================================

print("\n" + "="*80)
print("8. FEATURE RECOMMENDATIONS")
print("="*80)

# Identify features to remove
remove_recommendations = []

# 8.1 Constant columns (no information)
constant_cols = []
for col in df.columns:
    if df[col].nunique() == 1:
        constant_cols.append(col)
        remove_recommendations.append(f"{col} (constant value)")

# 8.2 Columns with >90% missing values
high_missing_cols = missing_df[missing_df['Missing %'] > 90]['Column'].tolist()
for col in high_missing_cols:
    remove_recommendations.append(f"{col} ({missing_df[missing_df['Column']==col]['Missing %'].iloc[0]:.1f}% missing)")

# 8.3 High cardinality columns that are likely IDs or URLs
high_card_id_cols = []
for col in high_cardinality['Column'].tolist():
    if 'id' in col.lower() or 'url' in col.lower() or 'desc' in col.lower():
        high_card_id_cols.append(col)
        remove_recommendations.append(f"{col} (high cardinality, likely identifier)")

# 8.4 Columns with all unique values (likely IDs)
all_unique_cols = []
for col in df.columns:
    if df[col].nunique() == len(df) and col not in ['id']:
        all_unique_cols.append(col)
        if col not in remove_recommendations:
            remove_recommendations.append(f"{col} (all unique values)")

# Print recommendations
print("\n🔴 RECOMMENDED TO REMOVE:")
if remove_recommendations:
    for rec in remove_recommendations[:20]:
        print(f"  • {rec}")
    if len(remove_recommendations) > 20:
        print(f"  ... and {len(remove_recommendations) - 20} more")
else:
    print("  No columns recommended for removal")

# 8.5 Features to keep (by category)
print("\n🟢 RECOMMENDED TO KEEP:")

# Core features (from your project design)
core_features = [
    'loan_amnt', 'annual_inc', 'dti', 'revol_util', 'emp_length', 
    'home_ownership', 'grade', 'sub_grade', 'purpose', 'term',
    'int_rate', 'installment', 'verification_status', 'addr_state',
    'fico_range_low', 'fico_range_high', 'inq_last_6mths', 
    'open_acc', 'revol_bal', 'total_acc'
]

# Check which core features exist
existing_core = [col for col in core_features if col in df.columns]
missing_core = [col for col in core_features if col not in df.columns]

print(f"\nCore features to keep (from project design):")
for col in existing_core:
    print(f"  • {col}")
if missing_core:
    print(f"\n  ⚠️ Missing core features: {missing_core}")

# Additional features with low missing values and reasonable cardinality
additional_features = []
for col in df.columns:
    if col not in core_features and col not in remove_recommendations:
        missing_pct = missing_df[missing_df['Column'] == col]['Missing %'].iloc[0] if col in missing_df['Column'].values else 0
        unique_count = df[col].nunique()
        if missing_pct < 20 and unique_count > 1 and unique_count < 5000:
            additional_features.append(col)

print(f"\nAdditional features worth exploring ({len(additional_features)}):")
for col in additional_features[:10]:
    print(f"  • {col}")
if len(additional_features) > 10:
    print(f"  ... and {len(additional_features) - 10} more")

# =============================================
# 9. Save EDA Report and Visualizations
# =============================================

print("\n" + "="*80)
print("9. SAVING EDA REPORT")
print("="*80)

report_path = os.path.join(OUTPUT_DIR, "eda_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("LENDING CLUB BASELINE EDA REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Total rows: {len(df):,}\n")
    f.write(f"Total columns: {len(df.columns)}\n\n")
    
    f.write("COLUMN TYPES:\n")
    f.write(f"  Numeric: {len(numeric_cols)}\n")
    f.write(f"  Categorical: {len(categorical_cols)}\n")
    f.write(f"  Object: {len(object_cols)}\n")
    f.write(f"  Datetime: {len(datetime_cols)}\n\n")
    
    f.write("MISSING VALUES:\n")
    f.write(f"  Columns with >50% missing: {len(high_missing)}\n")
    f.write(f"  Columns with 0 missing: {len(complete_cols)}\n\n")
    
    f.write("CARDINALITY:\n")
    f.write(f"  High cardinality columns (>1000): {len(high_cardinality)}\n")
    f.write(f"  Low cardinality columns (≤10): {len(low_cardinality)}\n\n")
    
    f.write("TARGET VARIABLE:\n")
    if 'loan_status' in df.columns:
        f.write(f"  Default rate: {default_mask.sum()/len(df)*100:.2f}%\n")
        f.write(f"  Defaults: {default_mask.sum():,}\n")
        f.write(f"  Non-defaults: {(~default_mask).sum():,}\n\n")
    
    f.write("RECOMMENDED TO REMOVE:\n")
    for rec in remove_recommendations:
        f.write(f"  • {rec}\n")
    
    f.write("\nRECOMMENDED TO KEEP:\n")
    for col in existing_core:
        f.write(f"  • {col}\n")
    
    f.write("\nADDITIONAL FEATURES TO EXPLORE:\n")
    for col in additional_features[:20]:
        f.write(f"  • {col}\n")

print(f"EDA report saved to: {report_path}")

# =============================================
# 10. Summary
# =============================================

print("\n" + "="*80)
print("10. EDA SUMMARY")
print("="*80)

print(f"\n✅ Analysis complete!")
print(f"  Total columns analyzed: {len(df.columns)}")
print(f"  Columns recommended to remove: {len(remove_recommendations)}")
print(f"  Core features to keep: {len(existing_core)}")
print(f"  Additional features to explore: {len(additional_features)}")
print(f"  Report saved to: {report_path}")
print(f"  Output directory: {OUTPUT_DIR}")

# List all saved visualizations
print("\n📊 Generated Visualizations:")
visualizations = [
    'column_types_distribution.png',
    'missing_values_heatmap.png',
    'missing_percentage_bar.png',
    'cardinality_distribution.png',
    'correlation_matrix.png',
    'loan_status_distribution.png',
    'default_distribution.png',
    'key_numeric_distributions.png'
]
for viz in visualizations:
    viz_path = os.path.join(OUTPUT_DIR, viz)
    if os.path.exists(viz_path):
        print(f"  • {viz}")

print("\nNext steps:")
print("  1. Remove identified redundant features")
print("  2. Keep core features for SSL pretraining")
print("  3. Create temporal split (2012-2016, 2017, 2018)")
print("  4. Start contrastive learning experiments")