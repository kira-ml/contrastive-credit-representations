"""
SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption
For tabular data - replaces random features with values from other samples
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class SCARFAugmentation:
    """
    SCARF augmentation for tabular data.
    For each sample, randomly selects a subset of features and replaces them
    with values from another random sample in the batch.
    """
    
    def __init__(self, corruption_rate: float = 0.3, num_corrupted_features: Optional[int] = None):
        """
        Args:
            corruption_rate: Fraction of features to corrupt (0.0 to 1.0)
            num_corrupted_features: Exact number of features to corrupt (overrides corruption_rate)
        """
        self.corruption_rate = corruption_rate
        self.num_corrupted_features = num_corrupted_features
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views using SCARF.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            x1, x2: Two augmented views with corrupted features
        """
        batch_size, n_features = x.shape
        
        # Determine number of features to corrupt
        if self.num_corrupted_features is not None:
            n_corrupt = min(self.num_corrupted_features, n_features)
        else:
            n_corrupt = int(n_features * self.corruption_rate)
        
        # Create two views
        x1 = x.clone()
        x2 = x.clone()
        
        # For each view, select random features and replace with values from random samples
        for view in [x1, x2]:
            # Select random features to corrupt
            corrupt_indices = torch.randperm(n_features)[:n_corrupt]
            
            # For each corrupted feature, select a random sample from the batch
            for feat_idx in corrupt_indices:
                # Randomly select replacement samples (excluding current sample)
                replacement_indices = torch.randint(0, batch_size, (batch_size,))
                # Ensure we don't use the same sample (optional)
                # view[:, feat_idx] = x[replacement_indices, feat_idx]
                view[:, feat_idx] = x[torch.randperm(batch_size), feat_idx]
        
        return x1, x2


class SCARFAugmentationV2:
    """
    SCARF V2 - More sophisticated version with:
    - Feature-wise corruption rates
    - Continuous vs categorical feature handling
    - Mixup for smoother augmentations
    """
    
    def __init__(self, 
                 continuous_corruption_rate: float = 0.2,
                 categorical_corruption_rate: float = 0.3,
                 use_mixup: bool = False,
                 mixup_alpha: float = 0.2):
        """
        Args:
            continuous_corruption_rate: Corruption rate for continuous features
            categorical_corruption_rate: Corruption rate for categorical features
            use_mixup: Whether to use mixup augmentation
            mixup_alpha: Alpha parameter for mixup
        """
        self.continuous_corruption_rate = continuous_corruption_rate
        self.categorical_corruption_rate = categorical_corruption_rate
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
    
    def __call__(self, x: torch.Tensor, continuous_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, n_features)
            continuous_mask: Boolean mask indicating continuous features (True) vs categorical (False)
            
        Returns:
            x1, x2: Augmented views
        """
        batch_size, n_features = x.shape
        
        x1 = x.clone()
        x2 = x.clone()
        
        # If no mask provided, treat all as continuous
        if continuous_mask is None:
            continuous_mask = torch.ones(n_features, dtype=torch.bool)
        
        # Split features
        continuous_indices = torch.where(continuous_mask)[0]
        categorical_indices = torch.where(~continuous_mask)[0]
        
        # Apply SCARF to continuous features
        n_cont_corrupt = int(len(continuous_indices) * self.continuous_corruption_rate)
        if n_cont_corrupt > 0 and len(continuous_indices) > 0:
            for view in [x1, x2]:
                corrupt_cont = continuous_indices[torch.randperm(len(continuous_indices))[:n_cont_corrupt]]
                for feat_idx in corrupt_cont:
                    view[:, feat_idx] = x[torch.randperm(batch_size), feat_idx]
        
        # Apply SCARF to categorical features
        n_cat_corrupt = int(len(categorical_indices) * self.categorical_corruption_rate)
        if n_cat_corrupt > 0 and len(categorical_indices) > 0:
            for view in [x1, x2]:
                corrupt_cat = categorical_indices[torch.randperm(len(categorical_indices))[:n_cat_corrupt]]
                for feat_idx in corrupt_cat:
                    view[:, feat_idx] = x[torch.randperm(batch_size), feat_idx]
        
        # Optional mixup augmentation
        if self.use_mixup:
            lambda_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x1 = lambda_mix * x1 + (1 - lambda_mix) * x1[torch.randperm(batch_size)]
            x2 = lambda_mix * x2 + (1 - lambda_mix) * x2[torch.randperm(batch_size)]
        
        return x1, x2