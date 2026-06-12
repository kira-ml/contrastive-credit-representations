"""
VICReg: Variance-Invariance-Covariance Regularization
Reference: https://arxiv.org/abs/2105.04906
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    """
    VICReg loss for self-supervised learning.
    Consists of three terms:
    - Variance: Keeps embeddings from collapsing
    - Invariance: Makes augmented views similar
    - Covariance: Reduces redundancy between dimensions
    """
    
    def __init__(self, 
                 invariance_weight: float = 25.0,
                 variance_weight: float = 25.0,
                 covariance_weight: float = 1.0,
                 variance_epsilon: float = 0.001):
        """
        Args:
            invariance_weight: Weight for invariance term
            variance_weight: Weight for variance term
            covariance_weight: Weight for covariance term
            variance_epsilon: Small epsilon for numerical stability
        """
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_epsilon = variance_epsilon
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute VICReg loss between two views.
        
        Args:
            z1: First view embeddings (batch_size, embedding_dim)
            z2: Second view embeddings (batch_size, embedding_dim)
            
        Returns:
            Total loss as a scalar tensor
        """
        batch_size, embedding_dim = z1.shape
        
        # ===== 1. INVARIANCE TERM =====
        # Mean squared error between the two views
        invariance_loss = F.mse_loss(z1, z2)
        
        # ===== 2. VARIANCE TERM =====
        # Compute standard deviation along batch dimension
        std_z1 = torch.sqrt(z1.var(dim=0) + self.variance_epsilon)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.variance_epsilon)
        
        # Hinge loss to keep variance above threshold
        variance_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
        
        # ===== 3. COVARIANCE TERM =====
        # Center the embeddings
        z1_centered = z1 - z1.mean(dim=0, keepdim=True)
        z2_centered = z2 - z2.mean(dim=0, keepdim=True)
        
        # Compute covariance matrices
        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)
        
        # Remove diagonal (variance) and take off-diagonal squared sum
        off_diag_z1 = cov_z1 - torch.diag_embed(torch.diag(cov_z1))
        off_diag_z2 = cov_z2 - torch.diag_embed(torch.diag(cov_z2))
        
        covariance_loss = off_diag_z1.pow(2).sum() / embedding_dim + off_diag_z2.pow(2).sum() / embedding_dim
        
        # ===== 4. TOTAL LOSS =====
        total_loss = (
            self.invariance_weight * invariance_loss +
            self.variance_weight * variance_loss +
            self.covariance_weight * covariance_loss
        )
        
        return total_loss


class VICRegLossV2(nn.Module):
    """
    VICReg loss with improved stability and batch normalization.
    """
    
    def __init__(self,
                 invariance_weight: float = 25.0,
                 variance_weight: float = 25.0,
                 covariance_weight: float = 1.0,
                 variance_epsilon: float = 0.001,
                 use_batch_norm: bool = True):
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_epsilon = variance_epsilon
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(64, affine=False)  # Match projection_dim
    
    def forward(self, z1, z2):
        if self.use_batch_norm:
            z1 = self.bn(z1)
            z2 = self.bn(z2)
        
        batch_size, embedding_dim = z1.shape
        
        # Invariance
        invariance_loss = F.mse_loss(z1, z2)
        
        # Variance
        std_z1 = torch.sqrt(z1.var(dim=0) + self.variance_epsilon)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.variance_epsilon)
        variance_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
        
        # Covariance
        z1_centered = z1 - z1.mean(dim=0, keepdim=True)
        z2_centered = z2 - z2.mean(dim=0, keepdim=True)
        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)
        off_diag_z1 = cov_z1 - torch.diag_embed(torch.diag(cov_z1))
        off_diag_z2 = cov_z2 - torch.diag_embed(torch.diag(cov_z2))
        covariance_loss = off_diag_z1.pow(2).sum() / embedding_dim + off_diag_z2.pow(2).sum() / embedding_dim
        
        total_loss = (
            self.invariance_weight * invariance_loss +
            self.variance_weight * variance_loss +
            self.covariance_weight * covariance_loss
        )
        
        return total_loss