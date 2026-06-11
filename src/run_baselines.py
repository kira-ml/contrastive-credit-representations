"""
Minimal Baseline Experiments for Credit Default Prediction
Compares: Logistic Regression, LightGBM, and PCA + LR
Uses preprocessed features from baseline_features_v2.npz
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================

DATA_PATH = r"D:\contrastive-credit-representations\data\processed\baseline_features_v2.npz"
N_SAMPLES = 50000  # Use subset for faster iteration
N_SEEDS = 5  # Number of random seeds for few-shot
FEW_SHOT_SIZES = [20, 50, 100, 200]

# =============================================
# Load Data
# =============================================

def load_data():
    """Load preprocessed features from .npz file"""
    data = np.load(DATA_PATH, allow_pickle=True)
    
    X_cont = data['X_continuous']
    X_cat = data['X_categorical']
    y = data['y']
    
    # Combine continuous and categorical
    X = np.hstack([X_cont, X_cat])
    
    return X, y

# =============================================
# Baseline Models
# =============================================

class BaselineModels:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'pca_lr': None  # Will be created with PCA
        }
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train all models and return metrics"""
        results = {}
        
        # 1. Logistic Regression
        lr = self.models['logistic_regression']
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict_proba(X_test)[:, 1]
        results['lr_auc'] = roc_auc_score(y_test, y_pred_lr)
        results['lr_f1'] = f1_score(y_test, lr.predict(X_test))
        
        # 2. LightGBM
        lgbm = self.models['lightgbm']
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict_proba(X_test)[:, 1]
        results['lgbm_auc'] = roc_auc_score(y_test, y_pred_lgbm)
        results['lgbm_f1'] = f1_score(y_test, lgbm.predict(X_test))
        
        # 3. PCA + Logistic Regression
        pca = PCA(n_components=0.95)  # Keep 95% variance
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        lr_pca = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_pca.fit(X_train_pca, y_train)
        y_pred_pca = lr_pca.predict_proba(X_test_pca)[:, 1]
        results['pca_lr_auc'] = roc_auc_score(y_test, y_pred_pca)
        results['pca_lr_f1'] = f1_score(y_test, lr_pca.predict(X_test_pca))
        
        return results

# =============================================
# Few-Shot Evaluation
# =============================================

def few_shot_evaluation(X, y, n_samples):
    """Run few-shot evaluation for given sample size"""
    
    # Stratified sampling to maintain class balance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    
    for seed in range(N_SEEDS):
        # Set random state for reproducibility
        np.random.seed(seed)
        
        # Sample few-shot training set
        # We want n_samples total with balanced classes
        n_defaults = n_samples // 2
        n_non_defaults = n_samples - n_defaults
        
        default_idx = np.where(y == 1)[0]
        non_default_idx = np.where(y == 0)[0]
        
        if len(default_idx) < n_defaults or len(non_default_idx) < n_non_defaults:
            print(f"⚠️ Not enough samples for n={n_samples}")
            continue
        
        # Sample randomly
        sampled_default = np.random.choice(default_idx, n_defaults, replace=False)
        sampled_non_default = np.random.choice(non_default_idx, n_non_defaults, replace=False)
        
        train_idx = np.concatenate([sampled_default, sampled_non_default])
        test_idx = np.random.choice(
            np.setdiff1d(np.arange(len(y)), train_idx), 
            size=min(5000, len(y) - len(train_idx)),
            replace=False
        )
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Train and evaluate
        models = BaselineModels()
        results = models.train_and_evaluate(X_train, y_train, X_test, y_test)
        results['seed'] = seed
        results['n_samples'] = n_samples
        
        all_results.append(results)
    
    return pd.DataFrame(all_results)

# =============================================
# Main Experiment
# =============================================

def run_experiments():
    """Run all few-shot experiments"""
    print("="*60)
    print("BASELINE EXPERIMENTS FOR CREDIT DEFAULT PREDICTION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X, y = load_data()
    
    # Subset for speed (use first N_SAMPLES)
    X = X[:N_SAMPLES]
    y = y[:N_SAMPLES]
    
    print(f"  Data shape: {X.shape}")
    print(f"  Default rate: {y.mean():.2%}")
    
    # Run experiments for each few-shot size
    all_results = []
    for n in FEW_SHOT_SIZES:
        print(f"\n{'='*40}")
        print(f"Few-Shot Size: {n} samples")
        print(f"{'='*40}")
        
        results_df = few_shot_evaluation(X, y, n)
        
        if len(results_df) > 0:
            # Calculate mean and std for each metric
            mean_results = results_df.groupby('n_samples').mean()
            std_results = results_df.groupby('n_samples').std()
            
            print(f"\nResults (mean ± std over {N_SEEDS} seeds):")
            print(f"{'Metric':<20} {'Mean':>10} {'Std':>10}")
            print("-" * 40)
            
            for metric in ['lr_auc', 'lgbm_auc', 'pca_lr_auc']:
                mean = mean_results[metric].values[0]
                std = std_results[metric].values[0]
                print(f"{metric:<20} {mean:>10.4f} {std:>10.4f}")
            
            all_results.append(results_df)
    
    # Combine all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Save results
        results_path = r"D:\contrastive-credit-representations\results\baselines\baseline_results.csv"
        final_results.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to: {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY TABLE (AUC)")
        print("="*60)
        
        summary = final_results.groupby('n_samples')[['lr_auc', 'lgbm_auc', 'pca_lr_auc']].agg(['mean', 'std'])
        print(summary)
        
        # Identify best model at each sample size
        print("\n" + "="*60)
        print("BEST MODEL AT EACH SAMPLE SIZE")
        print("="*60)
        
        for n in FEW_SHOT_SIZES:
            subset = final_results[final_results['n_samples'] == n]
            means = subset[['lr_auc', 'lgbm_auc', 'pca_lr_auc']].mean()
            best = means.idxmax()
            best_val = means.max()
            print(f"n={n:3d}: {best:<15} (AUC={best_val:.4f})")
    
    print("\n✅ Baseline experiments complete!")
    return final_results

# =============================================
# Quick Start
# =============================================

if __name__ == "__main__":
    results = run_experiments()