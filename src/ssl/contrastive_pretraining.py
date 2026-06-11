"""
Production-grade SimCLR Contrastive Pretraining for Credit Default Prediction
- Modular architecture
- Configurable via dataclasses
- Proper logging and checkpointing
- Error handling and validation
"""

import os
import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================

@dataclass
class SSLConfig:
    """Configuration for SSL pretraining"""
    # Data
    data_path: str = "D:/contrastive-credit-representations/data/processed/baseline_features_v2.npz"
    n_samples: int = 10000  # Use subset for pretraining
    batch_size: int = 256
    
    # Model architecture
    input_dim: int = 18  # Will be auto-detected
    hidden_dim: int = 128
    embedding_dim: int = 64
    projection_dim: int = 64  # For contrastive head
    
    # Training hyperparameters
    temperature: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 10
    warmup_epochs: int = 2
    
    # Augmentation
    noise_std: float = 0.1
    mask_prob: float = 0.2
    feature_dropout: float = 0.1
    
    # Checkpointing
    save_dir: str = "D:/contrastive-credit-representations/models"
    checkpoint_name: str = "ssl_encoder.pt"
    save_every: int = 5  # Save checkpoint every N epochs
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0  # Set to 0 for Windows compatibility
    seed: int = 42
    
    # Logging
    log_dir: str = "D:/contrastive-credit-representations/logs/ssl"
    log_level: str = "INFO"

# =============================================
# Logging Setup
# =============================================

def setup_logging(config: SSLConfig) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('ssl_pretraining')
    logger.setLevel(getattr(logging, config.log_level))
    
    # File handler
    log_file = log_dir / "pretraining.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# =============================================
# Data Loading
# =============================================

class CreditDataset(Dataset):
    """Dataset for contrastive pretraining"""
    
    def __init__(self, X: np.ndarray, n_samples: Optional[int] = None, seed: int = 42):
        """
        Args:
            X: Feature matrix
            n_samples: Number of samples to use (None for all)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        self.X = torch.FloatTensor(X)
        self.n_features = X.shape[1]
        
        if n_samples and n_samples < len(self.X):
            # Random subset for faster training
            idx = np.random.choice(len(self.X), n_samples, replace=False)
            self.X = self.X[idx]
            self.n_samples = n_samples
        else:
            self.n_samples = len(self.X)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]

def load_data(data_path: str, logger: logging.Logger, n_samples: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load preprocessed features from .npz file
    
    Returns:
        X: Feature matrix
        input_dim: Number of features
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    X_cont = data['X_continuous']
    X_cat = data['X_categorical']
    
    # Combine continuous and categorical features
    X = np.hstack([X_cont, X_cat])

    if np.isnan(X).any():
        logger.warning("NaN values detected in input data!")
        X = np.nan_to_num(X, nan=0.0)
    
    if n_samples and n_samples < len(X):
        np.random.seed(42)
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
    
    return X, X.shape[1]

# =============================================
# Augmentation for Tabular Data
# =============================================

class TabularAugmentation:
    """Production-grade tabular data augmentations"""
    
    def __init__(self, noise_std: float = 0.1, mask_prob: float = 0.2, 
                 feature_dropout: float = 0.1):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.feature_dropout = feature_dropout
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views of the input
        
        Args:
            x: Input tensor (batch_size, n_features)
        
        Returns:
            x1, x2: Augmented views
        """
        # Initialize both views
        x1 = x.clone()
        x2 = x.clone()
        
        # 1. Gaussian noise augmentation
        noise1 = torch.randn_like(x1) * self.noise_std
        noise2 = torch.randn_like(x2) * self.noise_std
        x1 = x1 + noise1
        x2 = x2 + noise2
        
        # 2. Feature masking (random dropout of features)
        mask1 = torch.rand_like(x1) > self.mask_prob
        mask2 = torch.rand_like(x2) > self.mask_prob
        x1 = x1 * mask1
        x2 = x2 * mask2
        
        # 3. Feature dropout (randomly zero out entire columns)
        if self.feature_dropout > 0:
            dropout_mask1 = torch.rand(x1.shape[1]) > self.feature_dropout
            dropout_mask2 = torch.rand(x2.shape[1]) > self.feature_dropout
            x1 = x1 * dropout_mask1.unsqueeze(0)
            x2 = x2 * dropout_mask2.unsqueeze(0)
        
        return x1, x2

# =============================================
# Encoder Architecture
# =============================================

class MLPEncoder(nn.Module):
    """Production-grade MLP encoder with residual connections"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 embedding_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Encoder with residual connections
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
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder"""
        return self.encoder(x)

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim: int, projection_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

# =============================================
# Contrastive Loss (SimCLR)
# =============================================

class SimCLRLoss(nn.Module):
    """NT-Xent loss for SimCLR (standard implementation)"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        # Normalize embeddings
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        batch_size = z1.shape[0]
        device = z1.device
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, embedding_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature
        
        # Create mask for positive pairs (diagonal of each block)
        pos_mask = torch.eye(2 * batch_size, device=device).to(torch.bool)
        
        # Apply the mask to get the logits for positive pairs
        pos_logits = sim_matrix[pos_mask].view(2 * batch_size, 1)
        
        # Create negative mask: all except the diagonal
        neg_mask = ~pos_mask
        
        # Get the logits for negative pairs
        neg_logits = sim_matrix[neg_mask].view(2 * batch_size, -1)
        
        # Concatenate positive and negative logits to form the full logit matrix
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        
        # Create labels: the first column is always the positive pair (index 0)
        labels = torch.zeros(2 * batch_size, device=device, dtype=torch.long)
        
        # Compute loss using the reshaped logits
        loss = nn.functional.cross_entropy(logits, labels)
        
        return loss



# =============================================
# Trainer
# =============================================

class SSLTrainer:
    """Production-grade SSL trainer"""
    
    def __init__(self, config: SSLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize components
        self.encoder = None
        self.projection_head = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.logger.info(f"Initialized SSL trainer on {self.device}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _init_model(self, input_dim: int):
        """Initialize encoder and projection head"""
        self.encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.embedding_dim
        ).to(self.device)
        
        self.projection_head = ProjectionHead(
            input_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim
        ).to(self.device)
        
        self.criterion = SimCLRLoss(temperature=self.config.temperature)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Log model architecture
        n_params = sum(p.numel() for p in self.encoder.parameters())
        self.logger.info(f"Encoder: {n_params:,} parameters")
        self.logger.info(f"Input dim: {input_dim} -> Hidden: {self.config.hidden_dim} -> Embedding: {self.config.embedding_dim}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.encoder.train()
        self.projection_head.train()
        
        total_loss = 0.0
        n_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            # Apply augmentations
            augment = TabularAugmentation(
                noise_std=0.05,  # Reduced from 0.1
                mask_prob=0.1,   # Reduced from 0.2
                feature_dropout=0.05  # Reduced from 0.1
            )
            x1, x2 = augment(batch)
            
            # Forward pass
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            
            # Projection head
            p1 = self.projection_head(z1)
            p2 = self.projection_head(z2)
            
            # Compute loss
            loss = self.criterion(p1, p2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.projection_head.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.debug(f"  Batch {batch_idx}/{n_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / n_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate (check for collapse)"""
        self.encoder.eval()
        
        # Sample embeddings and check variance
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                z = self.encoder(batch)
                embeddings.append(z.cpu())
        
        embeddings = torch.cat(embeddings, dim=0)
        variance = embeddings.std(dim=0).mean().item()
        
        return variance
    
    def train(self, dataloader: DataLoader, val_dataloader: DataLoader = None):
        """Full training loop"""
        self.logger.info("Starting SSL pretraining...")
        
        best_loss = float('inf')
        best_variance = 0.0
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss = self.train_epoch(dataloader, epoch)
            self.scheduler.step()
            
            # Validate
            if val_dataloader:
                variance = self.validate(val_dataloader)
                self.logger.info(f"Epoch {epoch}/{self.config.epochs} - Loss: {train_loss:.4f}, Variance: {variance:.4f}")
                
                # Check for collapse
                if variance < 0.01:
                    self.logger.warning(f"⚠️ Potential collapse detected! Variance: {variance:.4f}")
            else:
                self.logger.info(f"Epoch {epoch}/{self.config.epochs} - Loss: {train_loss:.4f}")
            
            # Save checkpoint
            if epoch % self.config.save_every == 0 or epoch == self.config.epochs:
                self.save_checkpoint(epoch, train_loss, variance)
            
            # Update best
            if train_loss < best_loss:
                best_loss = train_loss
                best_variance = variance
        
        self.logger.info("SSL pretraining complete!")
        self.logger.info(f"  Best loss: {best_loss:.4f}")
        self.logger.info(f"  Final variance: {best_variance:.4f}")
        
        # Save final model
        self.save_checkpoint(self.config.epochs, best_loss, best_variance, final=True)
    
    def save_checkpoint(self, epoch: int, loss: float, variance: float, final: bool = False):
        """Save model checkpoint"""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            checkpoint_path = save_dir / "ssl_encoder_final.pt"
        else:
            checkpoint_path = save_dir / f"ssl_encoder_epoch{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'variance': variance,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"  Saved checkpoint: {checkpoint_path}")
        
        # Also save encoder only for easy loading
        if final:
            encoder_path = save_dir / "ssl_encoder.pt"
            torch.save(self.encoder.state_dict(), encoder_path)
            self.logger.info(f"  Saved encoder only: {encoder_path}")

# =============================================
# Main Function
# =============================================

def main():
    """Main execution function"""
    
    # Load configuration
    config = SSLConfig()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("="*60)
    logger.info("SIMCLR CONTRASTIVE PRETRAINING")
    logger.info("="*60)
    
    # Log configuration
    logger.info("\nConfiguration:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Load data
        logger.info("\nLoading data...")
        X, input_dim = load_data(config.data_path, config.n_samples)
        logger.info(f"  Data shape: {X.shape}")
        logger.info(f"  Input dimension: {input_dim}")
        
        # Update config
        config.input_dim = input_dim
        
        # Create datasets
        dataset = CreditDataset(X)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  Batches per epoch: {len(train_loader)}")
        
        # Initialize trainer
        trainer = SSLTrainer(config, logger)
        trainer._init_model(input_dim)
        
        # Train
        trainer.train(train_loader, val_loader)
        
        logger.info("\nSSL pretraining completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during pretraining: {str(e)}")
        raise

# =============================================
# Entry Point
# =============================================

if __name__ == "__main__":
    main()