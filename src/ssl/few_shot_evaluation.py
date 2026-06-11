"""
Few-Shot Evaluation for SSL Pretrained Encoder
- Loads pretrained encoder from contrastive_pretraining.py
- Freezes encoder and trains linear probe on frozen embeddings
- Compares against Logistic Regression baseline
- Saves results to results/ssl/ssl_results.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================

DATA_PATH = r"D:\contrastive-credit-representations\data\processed\baseline_features_v2.npz"
ENCODER_PATH = r"D:\contrastive-credit-representations\models\ssl_encoder.pt"
RESULTS_DIR = r"D:\contrastive-credit-representations\results\ssl"
N_SAMPLES = 10000  # Use same subset as pretraining
BATCH_SIZE = 256
FEW_SHOT_SIZES = [20, 50, 100, 200]
N_SEEDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================
# Data Loading
# =============================================

def load_data(data_path: str, n_samples: int = None) -> tuple:
    """Load preprocessed features and labels"""
    data = np.load(data_path, allow_pickle=True)
    X_cont = data['X_continuous']
    X_cat = data['X_categorical']
    y = data['y']
    
    # Combine continuous and categorical
    X = np.hstack([X_cont, X_cat])
    
    if n_samples and n_samples < len(X):
        np.random.seed(42)
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]
    
    return X, y

# =============================================
# Encoder Model (Same architecture as pretraining)
# =============================================

class MLPEncoder(nn.Module):
    """MLP encoder matching pretraining architecture"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# =============================================
# SSL Embedding Extractor
# =============================================

class SSLFeatureExtractor:
    """Extracts frozen embeddings from pretrained encoder"""
    def __init__(self, encoder_path: str, input_dim: int):
        self.device = DEVICE
        self.encoder = MLPEncoder(input_dim=input_dim).to(self.device)
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.encoder.eval()
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def extract(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from features"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            embeddings = self.encoder(X_tensor).cpu().numpy()
        return embeddings

# =============================================
# Few-Shot Evaluation
# =============================================

def evaluate_few_shot(X: np.ndarray, y: np.ndarray, n_samples: int, seed: int) -> dict:
    """Evaluate SSL probe vs LR baseline for given sample size"""
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Stratified sampling to maintain class balance
    n_defaults = n_samples // 2
    n_non_defaults = n_samples - n_defaults
    
    default_idx = np.where(y == 1)[0]
    non_default_idx = np.where(y == 0)[0]
    
    # Check if we have enough samples
    if len(default_idx) < n_defaults or len(non_default_idx) < n_non_defaults:
        return None
    
    # Sample training set
    sampled_default = np.random.choice(default_idx, n_defaults, replace=False)
    sampled_non_default = np.random.choice(non_default_idx, n_non_defaults, replace=False)
    train_idx = np.concatenate([sampled_default, sampled_non_default])
    
    # Sample test set (stratified, 5000 samples max)
    remaining_idx = np.setdiff1d(np.arange(len(y)), train_idx)
    remaining_default = np.intersect1d(remaining_idx, default_idx)
    remaining_non_default = np.intersect1d(remaining_idx, non_default_idx)
    
    # Stratified test sampling
    n_test_default = min(2500, len(remaining_default))
    n_test_non_default = min(2500, len(remaining_non_default))
    
    test_default = np.random.choice(remaining_default, n_test_default, replace=False)
    test_non_default = np.random.choice(remaining_non_default, n_test_non_default, replace=False)
    test_idx = np.concatenate([test_default, test_non_default])
    
    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    results = {}
    
    # ===== 1. SSL Probe (Linear classifier on frozen embeddings) =====
    # Extract embeddings
    extractor = SSLFeatureExtractor(ENCODER_PATH, input_dim=X.shape[1])
    X_train_ssl = extractor.extract(X_train)
    X_test_ssl = extractor.extract(X_test)
    
    # Train linear classifier on embeddings
    ssl_probe = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        random_state=seed,
        class_weight='balanced'  # Handle class imbalance
    )
    ssl_probe.fit(X_train_ssl, y_train)
    y_pred_ssl = ssl_probe.predict_proba(X_test_ssl)[:, 1]
    
    results['ssl_auc'] = roc_auc_score(y_test, y_pred_ssl)
    results['ssl_f1'] = f1_score(y_test, ssl_probe.predict(X_test_ssl))
    
    # ===== 2. Logistic Regression Baseline (on raw features) =====
    lr_baseline = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        random_state=seed,
        class_weight='balanced'
    )
    lr_baseline.fit(X_train, y_train)
    y_pred_lr = lr_baseline.predict_proba(X_test)[:, 1]
    
    results['lr_auc'] = roc_auc_score(y_test, y_pred_lr)
    results['lr_f1'] = f1_score(y_test, lr_baseline.predict(X_test))
    
    # ===== 3. Compute SSL improvement =====
    results['ssl_improvement'] = results['ssl_auc'] - results['lr_auc']
    
    # Metadata
    results['n_samples'] = n_samples
    results['seed'] = seed
    results['train_size'] = len(train_idx)
    results['test_size'] = len(test_idx)
    
    return results

# =============================================
# Main Evaluation Loop
# =============================================

def run_few_shot_evaluation():
    """Run few-shot evaluation for all sample sizes"""
    print("="*60)
    print("FEW-SHOT EVALUATION - SSL PROBE VS BASELINE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X, y = load_data(DATA_PATH, n_samples=N_SAMPLES)
    print(f"  Data shape: {X.shape}")
    print(f"  Default rate: {y.mean():.2%}")
    
    # Verify encoder exists
    if not Path(ENCODER_PATH).exists():
        print(f"\n❌ Error: Encoder not found at {ENCODER_PATH}")
        print("  Please run contrastive_pretraining.py first")
        return
    
    # Run evaluation for each few-shot size
    all_results = []
    
    for n in FEW_SHOT_SIZES:
        print(f"\n{'='*40}")
        print(f"Few-Shot Size: {n} samples")
        print(f"{'='*40}")
        
        for seed in range(N_SEEDS):
            results = evaluate_few_shot(X, y, n, seed)
            
            if results is None:
                print(f"  ⚠️ Seed {seed}: Not enough samples, skipping")
                continue
            
            all_results.append(results)
            print(f"  Seed {seed}: SSL AUC = {results['ssl_auc']:.4f}, LR AUC = {results['lr_auc']:.4f}")
        
        # Print summary for this sample size
        df_n = pd.DataFrame([r for r in all_results if r['n_samples'] == n])
        if len(df_n) > 0:
            print(f"\n  Summary for n={n}:")
            print(f"    SSL AUC: {df_n['ssl_auc'].mean():.4f} ± {df_n['ssl_auc'].std():.4f}")
            print(f"    LR AUC:  {df_n['lr_auc'].mean():.4f} ± {df_n['lr_auc'].std():.4f}")
            print(f"    SSL Improvement: {df_n['ssl_improvement'].mean():.4f} ± {df_n['ssl_improvement'].std():.4f}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Create results directory
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        results_path = Path(RESULTS_DIR) / "ssl_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to: {results_path}")
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY TABLE")
        print("="*60)
        
        summary = results_df.groupby('n_samples')[['ssl_auc', 'lr_auc', 'ssl_improvement']].agg(['mean', 'std'])
        print(summary)
        
        # Print best improvement
        best_n = results_df.groupby('n_samples')['ssl_improvement'].mean().idxmax()
        best_imp = results_df.groupby('n_samples')['ssl_improvement'].mean().max()
        print(f"\nBest SSL improvement at n={best_n}: +{best_imp:.4f} AUC")
        
    else:
        print("\n❌ No results generated. Check data and encoder.")
    
    print("\n✅ Few-shot evaluation complete!")

# =============================================
# Entry Point
# =============================================

if __name__ == "__main__":
    run_few_shot_evaluation()