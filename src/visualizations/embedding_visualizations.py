"""
Eye-Catching Embedding Visualization for SSL Pretrained Encoder
Modern, professional styling with publication-ready themes
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import umap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================

DATA_PATH = r"D:\contrastive-credit-representations\data\processed\baseline_features_v2.npz"
ENCODER_PATH = r"D:\contrastive-credit-representations\models\ssl_encoder.pt"
OUTPUT_DIR = r"D:\contrastive-credit-representations\results\visualizations"
N_SAMPLES = 5000  # Number of samples to visualize
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================
# Modern Theme Configuration
# =============================================

# Color palettes
PALETTES = {
    'default': {
        'non_default': '#4ECDC4',  # Teal
        'default': '#FF6B6B',      # Coral
        'background': '#F8F9FA',    # Light gray
        'grid': '#E9ECEF',
        'text': '#2D3436'
    },
    'gradient': {
        'non_default': '#2ECC71',  # Emerald
        'default': '#E74C3C',      # Red
        'background': '#F8F9FA',
        'grid': '#E9ECEF',
        'text': '#2D3436'
    },
    'dark': {
        'non_default': '#00B894',  # Mint
        'default': '#FD79A8',      # Pink
        'background': '#2D3436',   # Dark
        'grid': '#636E72',
        'text': '#DFE6E9'
    }
}

# Choose theme
THEME = 'gradient'
COLORS = PALETTES[THEME]

# Modern font settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'grid.color': '#E9ECEF',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'axes.grid': True,
    'axes.labelcolor': '#2D3436',
    'xtick.color': '#2D3436',
    'ytick.color': '#2D3436',
})

# Seaborn theme
sns.set_theme(style='whitegrid', palette='viridis')

# =============================================
# Encoder Model
# =============================================

class MLPEncoder(nn.Module):
    """MLP encoder matching pretraining architecture"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 embedding_dim: int = 128, dropout: float = 0.1):
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
# Load Data and Embeddings
# =============================================

def load_data():
    """Load data and extract embeddings"""
    # Load data
    data = np.load(DATA_PATH, allow_pickle=True)
    X_cont = data['X_continuous']
    X_cat = data['X_categorical']
    y = data['y']
    
    # Combine features
    X = np.hstack([X_cont, X_cat])
    
    # Sample for visualization
    np.random.seed(42)
    idx = np.random.choice(len(X), N_SAMPLES, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]
    
    # Load encoder
    encoder = MLPEncoder(input_dim=X.shape[1]).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    
    # Extract embeddings
    X_tensor = torch.FloatTensor(X_sample).to(DEVICE)
    with torch.no_grad():
        embeddings = encoder(X_tensor).cpu().numpy()
    
    return embeddings, y_sample

# =============================================
# Visualization Functions (Enhanced Styling)
# =============================================

def plot_tsne(embeddings, y):
    """Generate t-SNE visualization with enhanced styling"""
    print("Generating t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot non-defaults with custom styling
    ax.scatter(embeddings_2d[y==0, 0], embeddings_2d[y==0, 1],
               c=COLORS['non_default'], alpha=0.8, s=15, 
               edgecolors='white', linewidth=0.5, label='Non-Default')
    
    # Plot defaults with custom styling
    ax.scatter(embeddings_2d[y==1, 0], embeddings_2d[y==1, 1],
               c=COLORS['default'], alpha=0.8, s=15, 
               edgecolors='white', linewidth=0.5, label='Default')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title('t-SNE of SSL Embeddings\n(colored by default status)', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Custom legend
    legend = ax.legend(loc='best', fontsize=12, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E9ECEF')
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'embeddings_tsne.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(OUTPUT_DIR) / 'embeddings_tsne.pdf', bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/embeddings_tsne.png")

def plot_umap(embeddings, y):
    """Generate UMAP visualization with enhanced styling"""
    print("Generating UMAP...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot non-defaults
    ax.scatter(embeddings_2d[y==0, 0], embeddings_2d[y==0, 1],
               c=COLORS['non_default'], alpha=0.8, s=15, 
               edgecolors='white', linewidth=0.5, label='Non-Default')
    
    # Plot defaults
    ax.scatter(embeddings_2d[y==1, 0], embeddings_2d[y==1, 1],
               c=COLORS['default'], alpha=0.8, s=15, 
               edgecolors='white', linewidth=0.5, label='Default')
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title('UMAP of SSL Embeddings\n(colored by default status)', 
                 fontsize=18, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='best', fontsize=12, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E9ECEF')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'embeddings_umap.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(OUTPUT_DIR) / 'embeddings_umap.pdf', bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/embeddings_umap.png")

def plot_silhouette(embeddings, y):
    """Plot silhouette scores with enhanced styling"""
    print("Generating silhouette analysis...")
    
    k_range = range(2, 11)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(k_range, silhouette_scores, 'o-', linewidth=3, 
            color='#2C3E50', markersize=10, markerfacecolor='#3498DB')
    ax.axvline(x=np.argmax(silhouette_scores) + 2, color='#E74C3C', 
               linestyle='--', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=14, fontweight='bold')
    ax.set_title('Silhouette Analysis for SSL Embeddings', 
                 fontsize=18, fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'silhouette_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(OUTPUT_DIR) / 'silhouette_analysis.pdf', bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/silhouette_analysis.png")

def plot_cluster_risk_profile(embeddings, y):
    """Show default rate by cluster with enhanced styling"""
    print("Generating cluster risk profile...")
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    default_rates = []
    for i in range(5):
        cluster_mask = cluster_labels == i
        default_rate = y[cluster_mask].mean()
        default_rates.append(default_rate)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(5)
    bars = ax.bar(x_pos, default_rates, color='#3498DB', alpha=0.8, 
                  edgecolor='white', linewidth=2)
    ax.axhline(y=y.mean(), color='#E74C3C', linestyle='--', 
               alpha=0.7, linewidth=2, label='Overall Default Rate')
    
    for bar, rate in zip(bars, default_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{rate:.2%}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Cluster', fontsize=14, fontweight='bold')
    ax.set_ylabel('Default Rate', fontsize=14, fontweight='bold')
    ax.set_title('Default Rate by Cluster in Embedding Space', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(5)], fontsize=12)
    ax.legend(loc='best', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'cluster_risk_profile.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(OUTPUT_DIR) / 'cluster_risk_profile.pdf', bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/cluster_risk_profile.png")

def plot_risk_score_distribution(embeddings, y):
    """Distribution of risk scores with enhanced styling"""
    print("Generating risk score distribution...")
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    
    default_rate_cluster0 = y[labels == 0].mean()
    default_rate_cluster1 = y[labels == 1].mean()
    
    if default_rate_cluster0 > default_rate_cluster1:
        default_center = centers[0]
    else:
        default_center = centers[1]
    
    distances = np.linalg.norm(embeddings - default_center, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot non-defaults with custom styling
    ax.hist(distances[y==0], bins=30, alpha=0.7, color=COLORS['non_default'], 
            edgecolor='white', linewidth=1, label='Non-Default')
    
    # Plot defaults with custom styling
    ax.hist(distances[y==1], bins=30, alpha=0.7, color=COLORS['default'], 
            edgecolor='white', linewidth=1, label='Default')
    
    ax.set_xlabel('Distance to Default Cluster Center', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title('Risk Score Distribution\n(Distance to Default Cluster)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'risk_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(OUTPUT_DIR) / 'risk_score_distribution.pdf', bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/risk_score_distribution.png")

def plot_feature_correlation(embeddings, y):
    """Heatmap of feature correlations with enhanced styling"""
    print("Generating feature correlation heatmap...")
    
    corr_matrix = np.corrcoef(embeddings.T)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Correlation', 'shrink': 0.8},
                ax=ax, square=True)
    
    ax.set_xlabel('Embedding Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('Embedding Dimension', fontsize=14, fontweight='bold')
    ax.set_title('Feature Correlation Matrix in Embedding Space', 
                 fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(OUTPUT_DIR) / 'feature_correlation.pdf', bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/feature_correlation.png")

def plot_nearest_neighbors(embeddings, y):
    """Visualize nearest neighbors with enhanced styling"""
    print("Generating nearest neighbors visualization...")
    
    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    label_consistency = []
    for i in range(len(embeddings)):
        neighbor_labels = y[indices[i]]
        consistency = (neighbor_labels == y[i]).mean()
        label_consistency.append(consistency)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.hist(label_consistency[y==0], bins=20, alpha=0.7, color=COLORS['non_default'], 
            edgecolor='white', linewidth=1, label='Non-Default')
    ax.hist(label_consistency[y==1], bins=20, alpha=0.7, color=COLORS['default'], 
            edgecolor='white', linewidth=1, label='Default')
    
    ax.axvline(x=0.5, color='#2C3E50', linestyle='--', 
               alpha=0.7, linewidth=2, label='Random')
    
    ax.set_xlabel('Proportion of Neighbors with Same Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title('Nearest Neighbor Label Consistency', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'nearest_neighbors.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(OUTPUT_DIR) / 'nearest_neighbors.pdf', bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/nearest_neighbors.png")

# =============================================
# Main Execution
# =============================================

def main():
    """Generate embedding visualizations"""
    print("="*60)
    print("EYE-CATCHING EMBEDDING VISUALIZATION")
    print("="*60)
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data and embeddings
    print("\nLoading data and extracting embeddings...")
    embeddings, y = load_data()
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Default rate: {y.mean():.2%}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_tsne(embeddings, y)
    plot_umap(embeddings, y)
    plot_silhouette(embeddings, y)
    plot_cluster_risk_profile(embeddings, y)
    plot_risk_score_distribution(embeddings, y)
    plot_feature_correlation(embeddings, y)
    plot_nearest_neighbors(embeddings, y)
    
    print("\n✅ Enhanced embedding visualizations complete!")
    print(f"  Output directory: {OUTPUT_DIR}")

# =============================================
# Entry Point
# =============================================

if __name__ == "__main__":
    main()