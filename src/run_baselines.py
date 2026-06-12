"""
Baseline Comparison Runner
Automatically loads all feature sets from data/processed/ and compares baseline models
Supports Parquet files with 'target' column
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================

DATA_DIR = Path(r"D:\contrastive-credit-representations\data\processed")
RESULTS_DIR = Path(r"D:\contrastive-credit-representations\results\baselines")
N_SAMPLES = 50000  # Use subset for speed
N_SEEDS = 5
FEW_SHOT_SIZES = [20, 50, 100, 200]

# =============================================
# Core Functions
# =============================================

def load_parquet_feature_set(filepath):
    """Load a single feature set from Parquet file"""
    df = pd.read_parquet(filepath)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    return X, y, df.columns.tolist()

def get_available_feature_sets():
    """Find all .parquet files in data/processed/"""
    files = list(DATA_DIR.glob("*.parquet"))
    # Exclude files that are just metadata or temp files
    files = [f for f in files if not f.name.startswith('temp_')]
    return sorted(files)

def evaluate_few_shot(X, y, n_samples, seed):
    """Evaluate all baselines on one few-shot sample"""
    np.random.seed(seed)
    
    # Stratified sampling
    n_defaults = n_samples // 2
    n_non_defaults = n_samples - n_defaults
    
    default_idx = np.where(y == 1)[0]
    non_default_idx = np.where(y == 0)[0]
    
    if len(default_idx) < n_defaults or len(non_default_idx) < n_non_defaults:
        return None
    
    # Sample training set
    train_default = np.random.choice(default_idx, n_defaults, replace=False)
    train_non_default = np.random.choice(non_default_idx, n_non_defaults, replace=False)
    train_idx = np.concatenate([train_default, train_non_default])
    
    # Sample test set (stratified, max 5000)
    remaining = np.setdiff1d(np.arange(len(y)), train_idx)
    remaining_default = np.intersect1d(remaining, default_idx)
    remaining_non_default = np.intersect1d(remaining, non_default_idx)
    
    n_test_default = min(2500, len(remaining_default))
    n_test_non_default = min(2500, len(remaining_non_default))
    
    test_default = np.random.choice(remaining_default, n_test_default, replace=False)
    test_non_default = np.random.choice(remaining_non_default, n_test_non_default, replace=False)
    test_idx = np.concatenate([test_default, test_non_default])
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    results = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict_proba(X_test)[:, 1]
    results['lr_auc'] = roc_auc_score(y_test, y_pred_lr)
    results['lr_f1'] = f1_score(y_test, lr.predict(X_test))
    
    # 2. LightGBM
    lgbm = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=seed, verbose=-1, class_weight='balanced'
    )
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict_proba(X_test)[:, 1]
    results['lgbm_auc'] = roc_auc_score(y_test, y_pred_lgbm)
    results['lgbm_f1'] = f1_score(y_test, lgbm.predict(X_test))
    
    # 3. PCA + Logistic Regression
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lr_pca = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    lr_pca.fit(X_train_pca, y_train)
    y_pred_pca = lr_pca.predict_proba(X_test_pca)[:, 1]
    results['pca_lr_auc'] = roc_auc_score(y_test, y_pred_pca)
    results['pca_lr_f1'] = f1_score(y_test, lr_pca.predict(X_test_pca))
    
    # Metadata
    results['n_samples'] = n_samples
    results['seed'] = seed
    
    return results

# =============================================
# Main Comparison Runner
# =============================================

def run_comparison(feature_sets=None, quick=False):
    """Run baseline comparison across multiple feature sets"""
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get feature sets to test
    if feature_sets is None:
        feature_sets = get_available_feature_sets()
    
    print("="*80)
    print("BASELINE MODEL COMPARISON ACROSS FEATURE SETS")
    print("="*80)
    print(f"\nTesting {len(feature_sets)} feature sets:")
    for f in feature_sets:
        print(f"  • {f.name}")
    
    all_results = []
    
    for filepath in feature_sets:
        print(f"\n{'='*60}")
        print(f"Testing: {filepath.name}")
        print(f"{'='*60}")
        
        # Load data
        X, y, columns = load_parquet_feature_set(filepath)
        
        # Load metadata if available
        meta_path = filepath.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"  Features: {metadata.get('n_features', len(columns)-1)}")
            print(f"  Default rate: {metadata.get('default_rate', y.mean()):.2%}")
        else:
            print(f"  Features: {len(columns)-1}")
            print(f"  Default rate: {y.mean():.2%}")
        
        # Subset for speed if needed
        if len(X) > N_SAMPLES:
            idx = np.random.RandomState(42).choice(len(X), N_SAMPLES, replace=False)
            X, y = X[idx], y[idx]
            print(f"  Using subset: {N_SAMPLES} samples")
        
        print(f"  Data shape: {X.shape}")
        
        # Run few-shot evaluation for each sample size
        for n in FEW_SHOT_SIZES:
            print(f"\n  Few-Shot Size: {n}")
            
            for seed in range(N_SEEDS):
                results = evaluate_few_shot(X, y, n, seed)
                if results is not None:
                    results['feature_set'] = filepath.name
                    all_results.append(results)
            
            # Print summary for this n
            df_n = pd.DataFrame([r for r in all_results if r['feature_set'] == filepath.name and r['n_samples'] == n])
            if len(df_n) > 0:
                print(f"    LR AUC: {df_n['lr_auc'].mean():.4f} ± {df_n['lr_auc'].std():.4f}")
                print(f"    LGBM AUC: {df_n['lgbm_auc'].mean():.4f} ± {df_n['lgbm_auc'].std():.4f}")
                print(f"    PCA+LR AUC: {df_n['pca_lr_auc'].mean():.4f} ± {df_n['pca_lr_auc'].std():.4f}")
        
        # Save intermediate results
        temp_df = pd.DataFrame([r for r in all_results if r['feature_set'] == filepath.name])
        temp_path = RESULTS_DIR / f"temp_{filepath.stem}.csv"
        temp_df.to_csv(temp_path, index=False)
        print(f"\n  ✅ Intermediate results saved to: {temp_path}")
    
    # Combine all results
    if all_results:
        final_df = pd.DataFrame(all_results)
        output_path = RESULTS_DIR / "baseline_comparison_results.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"✅ COMPARISON COMPLETE")
        print(f"   Results saved to: {output_path}")
        print(f"   Total evaluations: {len(final_df)}")
        print(f"{'='*80}")
        
        # Print summary table
        print("\nSUMMARY TABLE (AUC by feature set and sample size)")
        print("-" * 80)
        
        summary = final_df.groupby(['feature_set', 'n_samples'])[['lr_auc', 'lgbm_auc', 'pca_lr_auc']].mean().round(4)
        print(summary)
        
        # Identify winner
        print("\n" + "="*80)
        print("BEST FEATURE SET OVERALL (avg across all sample sizes)")
        print("="*80)
        
        overall = final_df.groupby('feature_set')[['lr_auc', 'lgbm_auc', 'pca_lr_auc']].mean()
        overall['avg_auc'] = overall.mean(axis=1)
        best = overall['avg_auc'].idxmax()
        print(f"\nWinner: {best}")
        print(f"  Avg AUC: {overall.loc[best, 'avg_auc']:.4f}")
        print("\nRanking:")
        for i, (name, row) in enumerate(overall.sort_values('avg_auc', ascending=False).iterrows(), 1):
            print(f"  {i:2d}. {name}: {row['avg_auc']:.4f}")
        
        # Save best feature set name
        winner_path = RESULTS_DIR / "best_feature_set.txt"
        with open(winner_path, 'w') as f:
            f.write(best)
        print(f"\n  ✅ Best feature set saved to: {winner_path}")
        
    else:
        print("\n❌ No results generated. Check data files.")
    
    return final_df

# =============================================
# CLI
# =============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run baseline comparison across feature sets')
    parser.add_argument('--feature_sets', nargs='+', help='Specific .parquet files to test (optional)')
    parser.add_argument('--quick', action='store_true', help='Run only N=50 and N=100')
    parser.add_argument('--n_samples', type=int, default=50000, help='Number of samples to use')
    
    args = parser.parse_args()
    
    if args.n_samples:
        N_SAMPLES = args.n_samples
    
    if args.quick:
        FEW_SHOT_SIZES = [50, 100]
    
    # If specific feature sets provided, convert to Path objects
    if args.feature_sets:
        feature_sets = [Path(DATA_DIR / f) for f in args.feature_sets]
    else:
        feature_sets = None
    
    run_comparison(feature_sets)