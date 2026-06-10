"""
Optimized CSV to Parquet converter for Intel Core i5 laptops
- Chunked processing to avoid memory issues
- Memory-efficient data types (float32, int32, category)
- Handles Lending Club footer rows with summary statistics
- Handles date columns properly
- Uses inspection results to handle all columns correctly
"""

import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================

# Input paths
ACCEPTED_CSV = r"D:\archive (28)\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv"
REJECTED_CSV = r"D:\archive (28)\rejected_2007_to_2018q4.csv\rejected_2007_to_2018Q4.csv"

# Output path
OUTPUT_DIR = r"D:\contrastive-credit-representations\data\raw"

# Performance settings for Core i5
CHUNK_SIZE = 200_000  # Optimal for 8-16 GB RAM
COMPRESSION = 'snappy'  # fastparquet works best with snappy

# =============================================
# Create output directory
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Chunk size: {CHUNK_SIZE:,} rows")
print("="*60)

# =============================================
# Helper functions
# =============================================

def optimize_dtypes(df):
    """Convert columns to memory-efficient dtypes"""
    
    # Convert float64 to float32 (saves 50% memory)
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype('float32')
    
    # Convert int64 to int32 (saves 50% memory)
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = df[col].astype('int32')
    
    # Convert object columns with low cardinality to category
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df[col].nunique() < 100:  # Low cardinality = category
            df[col] = df[col].astype('category')
        else:
            # For high cardinality, keep as object
            df[col] = df[col].astype('object')
    
    return df

def clean_chunk(chunk):
    """Remove footer rows and handle date columns"""
    
    # Check if 'id' column exists (for accepted loans)
    if 'id' in chunk.columns:
        # Convert id to string
        chunk['id'] = chunk['id'].astype(str)
        
        # Filter out rows where id doesn't look like a numeric ID
        # This removes footer rows like "Total amount funded..."
        chunk = chunk[chunk['id'].str.match(r'^\d+$', na=False)]
    
    # Check for 'Amount Requested' column (for rejected loans)
    if 'Amount Requested' in chunk.columns:
        # Remove rows where Amount Requested contains text
        chunk = chunk[pd.to_numeric(chunk['Amount Requested'], errors='coerce').notna()]
    
    # Also check for other columns that might have footer data
    if 'loan_amnt' in chunk.columns:
        # Remove rows where loan_amnt contains text
        chunk = chunk[pd.to_numeric(chunk['loan_amnt'], errors='coerce').notna()]
    
    # Handle date columns for accepted loans
    date_columns_accepted = ['earliest_cr_line', 'sec_app_earliest_cr_line', 'issue_d', 'last_pymnt_d', 
                             'last_credit_pull_d', 'next_pymnt_d', 'mths_since_last_delinq',
                             'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
                             'debt_settlement_flag_date', 'settlement_date']
    
    for col in date_columns_accepted:
        if col in chunk.columns:
            # Try multiple date formats
            try:
                chunk[col] = pd.to_datetime(chunk[col], errors='coerce', format='%b-%Y')
            except:
                try:
                    chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
                except:
                    chunk[col] = pd.NaT
            
            # Convert to string representation that fastparquet can handle
            if chunk[col].notna().any():
                chunk[col] = chunk[col].dt.strftime('%Y-%m-%d')
            else:
                chunk[col] = None
    
    # Handle date columns for rejected loans
    date_columns_rejected = ['Application Date']
    
    for col in date_columns_rejected:
        if col in chunk.columns:
            try:
                chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
            except:
                chunk[col] = pd.NaT
            
            # Convert to string representation
            if chunk[col].notna().any():
                chunk[col] = chunk[col].dt.strftime('%Y-%m-%d')
            else:
                chunk[col] = None
    
    return chunk

def process_chunk(chunk, output_path, is_first_chunk=False):
    """Process a single chunk and append to Parquet using fastparquet"""
    
    # Clean the chunk (remove footer rows and handle dates)
    chunk = clean_chunk(chunk)
    
    # Skip empty chunks
    if len(chunk) == 0:
        return 0
    
    # Optimize data types
    chunk = optimize_dtypes(chunk)
    
    if is_first_chunk:
        chunk.to_parquet(
            output_path,
            engine='fastparquet',
            compression=COMPRESSION,
            index=False
        )
    else:
        chunk.to_parquet(
            output_path,
            engine='fastparquet',
            compression=COMPRESSION,
            index=False,
            append=True
        )
    
    return len(chunk)

def convert_csv_to_parquet(csv_path, parquet_path, description="Converting"):
    """Convert CSV to Parquet with chunked processing"""
    
    print(f"\n{description}: {os.path.basename(csv_path)}")
    print(f"  Output: {os.path.basename(parquet_path)}")
    
    start_time = time.time()
    
    # Get total rows for progress (fast approximate)
    print("  Counting rows...", end=" ")
    with open(csv_path, 'rb') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    print(f"{total_rows:,}")
    
    # Process chunks
    print(f"  Processing {total_rows:,} rows in chunks of {CHUNK_SIZE:,}")
    
    chunk_iter = pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding='utf-8',
        on_bad_lines='skip'
    )
    
    rows_processed = 0
    chunk_count = 0
    valid_rows = 0
    
    for i, chunk in enumerate(chunk_iter):
        is_first = (i == 0)
        chunk_rows = process_chunk(chunk, parquet_path, is_first)
        rows_processed += len(chunk)
        valid_rows += chunk_rows
        chunk_count += 1
        
        # Progress update
        progress = (rows_processed / total_rows) * 100
        elapsed = time.time() - start_time
        speed = rows_processed / elapsed if elapsed > 0 else 0
        
        print(f"    Chunk {chunk_count:,}: {rows_processed:,} / {total_rows:,} rows "
              f"({progress:.1f}%) @ {speed:,.0f} rows/sec", end="\r")
    
    print(f"\n  ✅ {description} complete!")
    print(f"     Total rows processed: {rows_processed:,}")
    print(f"     Valid rows kept: {valid_rows:,}")
    print(f"     Footer rows removed: {rows_processed - valid_rows:,}")
    print(f"     Time: {time.time() - start_time:.2f} seconds")
    print(f"     Speed: {valid_rows / (time.time() - start_time):,.0f} rows/sec")
    
    # Show file sizes
    csv_size = os.path.getsize(csv_path) / 1024**3
    parquet_size = os.path.getsize(parquet_path) / 1024**3
    print(f"     CSV size: {csv_size:.2f} GB")
    print(f"     Parquet size: {parquet_size:.2f} GB")
    print(f"     Compression ratio: {csv_size / parquet_size:.2f}x")

# =============================================
# Convert accepted loans
# =============================================

accepted_output = os.path.join(OUTPUT_DIR, "accepted_2007_to_2018Q4.parquet")

# Delete existing file to ensure clean conversion
if os.path.exists(accepted_output):
    print(f"\n⚠️  Removing existing accepted Parquet file: {accepted_output}")
    os.remove(accepted_output)

convert_csv_to_parquet(ACCEPTED_CSV, accepted_output, "Converting accepted loans")

# =============================================
# Convert rejected loans
# =============================================

rejected_output = os.path.join(OUTPUT_DIR, "rejected_2007_to_2018Q4.parquet")

# Delete existing file to ensure clean conversion
if os.path.exists(rejected_output):
    print(f"\n⚠️  Removing existing rejected Parquet file: {rejected_output}")
    os.remove(rejected_output)

convert_csv_to_parquet(REJECTED_CSV, rejected_output, "Converting rejected loans")

# =============================================
# Final summary with error handling
# =============================================

print("\n" + "="*60)
print("CONVERSION COMPLETE")
print("="*60)

# Verify files - use fastparquet for all reads
accepted_parquet = os.path.join(OUTPUT_DIR, "accepted_2007_to_2018Q4.parquet")
rejected_parquet = os.path.join(OUTPUT_DIR, "rejected_2007_to_2018Q4.parquet")

if os.path.exists(accepted_parquet):
    try:
        accepted_df = pd.read_parquet(accepted_parquet, engine='fastparquet')
        print(f"✓ Accepted loans: {len(accepted_df):,} rows, {len(accepted_df.columns)} columns")
        print(f"  Memory: {accepted_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"⚠️  Could not read accepted loans file: {e}")

if os.path.exists(rejected_parquet):
    try:
        rejected_df = pd.read_parquet(rejected_parquet, engine='fastparquet')
        print(f"✓ Rejected loans: {len(rejected_df):,} rows, {len(rejected_df.columns)} columns")
        print(f"  Memory: {rejected_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"⚠️  Could not read rejected loans file: {e}")

print("\n✅ All conversions complete!")
print(f"Output directory: {OUTPUT_DIR}")