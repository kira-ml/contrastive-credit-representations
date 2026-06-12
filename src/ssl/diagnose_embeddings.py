import torch
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Load data
data_path = Path(r"D:\contrastive-credit-representations\data\processed\baseline_features_v2.npz")
data = np.load(data_path, allow_pickle=True)
X_cont = data['X_continuous']
X_cat = data['X_categorical']
y = data['y']
X = np.hstack([X_cont, X_cat])

# Load encoder
from contrastive_pretraining import MLPEncoder
encoder = MLPEncoder(input_dim=X.shape[1], hidden_dim=256, embedding_dim=128)
encoder_path = Path(r"D:\contrastive-credit-representations\models\ssl_encoder.pt")
encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
encoder.eval()

# Get embeddings
with torch.no_grad():
    X_tensor = torch.FloatTensor(X)
    embeddings = encoder(X_tensor).numpy()

# Sample for t-SNE
sample_size = min(5000, len(embeddings))
idx = np.random.choice(len(embeddings), sample_size, replace=False)
embeddings_sample = embeddings[idx]
y_sample = y[idx]

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_sample)

# Silhouette score
sil_score = silhouette_score(embeddings_sample, y_sample)
print(f"Silhouette score: {sil_score:.4f}")

# Check separation
default_emb = embeddings[y == 1]
non_default_emb = embeddings[y == 0]
print(f"Default embeddings mean: {default_emb.mean(axis=0).mean():.4f}")
print(f"Non-default embeddings mean: {non_default_emb.mean(axis=0).mean():.4f}")

# Print a few stats
print(f"Embedding variance: {embeddings.var():.4f}")