"""
Lightweight data inspection script for large Lending Club datasets
- Uses chunked reading to avoid memory issues
- Samples data intelligently
- Shows column types, null counts, and sample values
"""

import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

# =============================================
# Configuration
# =============================================

ACCEPTED_CSV = r"D:\archive (28)\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv"
REJECTED_CSV = r"D:\archive (28)\rejected_2007_to_2018q4.csv\rejected_2007_to_2018Q4.csv"

# Performance settings
SAMPLE_SIZE = 10000  # Number of rows to sample
CHUNK_SIZE = 50000   # Process in chunks to avoid memory issues

# =============================================
# Helper functions
# =============================================

def inspect_large_csv(csv_path, sample_size=10000, chunk_size=50000):
    """
    Inspect a large CSV file without loading it entirely into memory
    """
    print(f"\n{'='*60}")
    print(f"INSPECTING: {os.path.basename(csv_path)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 1. Get file size
    file_size = os.path.getsize(csv_path) / 1024**3
    print(f"File size: {file_size:.2f} GB")
    
    # 2. Count total rows (fast)
    print("Counting rows...", end=" ")
    with open(csv_path, 'rb') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    print(f"{total_rows:,}")
    
    # 3. Get column names from first row
    first_chunk = pd.read_csv(csv_path, nrows=1, low_memory=False)
    columns = first_chunk.columns.tolist()
    print(f"Total columns: {len(columns)}")
    
    # 4. Sample data to understand structure
    print(f"\nSampling {sample_size:,} rows (0.1-1% of data)...")
    
    # Read in chunks and collect a sample
    sample_rows = []
    chunk_iter = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        low_memory=False,
        encoding='utf-8',
        on_bad_lines='skip'
    )
    
    rows_read = 0
    for chunk in chunk_iter:
        sample_rows.append(chunk.sample(n=min(len(chunk), sample_size // 10)))
        rows_read += len(chunk)
        if rows_read >= sample_size * 10:  # Stop after reading enough
            break
    
    # Combine samples
    sample_df = pd.concat(sample_rows, ignore_index=True).sample(n=min(sample_size, len(sample_df)) if len(sample_df) > sample_size else sample_df)
    
    # 5. Display basic statistics
    print(f"\n--- BASIC STATISTICS ---")
    print(f"Sample shape: {sample_df.shape}")
    print(f"Memory usage of sample: {sample_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 6. Show first 5 rows
    print(f"\n--- FIRST 5 ROWS ---")
    print(sample_df.head())
    
    # 7. Show data types (first 20 columns)
    print(f"\n--- DATA TYPES (First 20 columns) ---")
    for col in columns[:20]:
        dtype = sample_df[col].dtype
        null_count = sample_df[col].isna().sum()
        unique_count = sample_df[col].nunique()
        print(f"  {col}: {dtype} | Nulls: {null_count}/{len(sample_df)} | Unique: {unique_count}")
    
    # 8. Detect date columns
    print(f"\n--- DATE COLUMNS DETECTION ---")
    date_cols = []
    for col in columns:
        if sample_df[col].dtype == 'object':
            sample_vals = sample_df[col].dropna().head(5)
            for val in sample_vals:
                if isinstance(val, str):
                    if any(m in val for m in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','-','/']):
                        date_cols.append(col)
                        break
    print(f"Date columns detected: {date_cols}")
    
    # 9. Show sample values for object columns
    print(f"\n--- SAMPLE VALUES FOR OBJECT COLUMNS ---")
    for col in columns:
        if sample_df[col].dtype == 'object':
            sample_vals = sample_df[col].dropna().head(3).tolist()
            print(f"  {col}: {sample_vals}")
    
    # 10. Check for footer rows
    print(f"\n--- CHECKING FOR FOOTER ROWS ---")
    # Read last 10 rows of the file
    with open(csv_path, 'rb') as f:
        f.seek(0, 2)  # Go to end
        file_size = f.tell()
        # Read last ~10KB
        f.seek(max(0, file_size - 10000))
        last_lines = f.read().decode('utf-8', errors='ignore').splitlines()
        last_10 = last_lines[-10:] if len(last_lines) >= 10 else last_lines
    print(f"Last 10 rows of file:")
    for i, line in enumerate(last_10):
        print(f"  {i+1}: {line[:100]}...")
    
    # 11. Summary
    print(f"\n--- SUMMARY ---")
    print(f"Total rows: {total_rows:,}")
    print(f"Total columns: {len(columns)}")
    print(f"Date columns: {len(date_cols)}")
    print(f"Inspection time: {time.time() - start_time:.2f} seconds")
    
    return {
        'total_rows': total_rows,
        'columns': columns,
        'date_columns': date_cols,
        'file_size': file_size,
        'sample_shape': sample_df.shape
    }

# =============================================
# Run inspection
# =============================================

if __name__ == "__main__":
    print("LENDING CLUB DATA INSPECTION")
    print("Optimized for Intel Core i5\n")
    
    # Inspect accepted loans
    accepted_info = inspect_large_csv(ACCEPTED_CSV, sample_size=10000)
    
    # Inspect rejected loans
    rejected_info = inspect_large_csv(REJECTED_CSV, sample_size=10000)
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)
    print(f"\nAccepted loans: {accepted_info['total_rows']:,} rows, {len(accepted_info['columns'])} columns")
    print(f"Rejected loans: {rejected_info['total_rows']:,} rows, {len(rejected_info['columns'])} columns")
    print(f"\nDate columns in accepted: {accepted_info['date_columns']}")
    print(f"Date columns in rejected: {rejected_info['date_columns']}")