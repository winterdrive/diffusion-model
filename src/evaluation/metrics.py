"""
Evaluation metrics for diffusion models.
"""

import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F


def accuracy_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate accuracy score for multi-label classification.
    
    Args:
        predictions: Predicted probabilities [batch_size, num_classes]
        targets: True binary labels [batch_size, num_classes]
        threshold: Classification threshold
    
    Returns:
        float: Accuracy score
    """
    pred_binary = (predictions > threshold).float()
    correct = (pred_binary == targets).float()
    accuracy = correct.mean().item()
    return accuracy


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and fake features.
    
    Args:
        real_features: Features from real images [N, feature_dim]
        fake_features: Features from generated images [M, feature_dim]
    
    Returns:
        float: FID score
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean = np.sqrt(sigma1.dot(sigma2))
    
    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def calculate_is(features: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS) for generated images.
    
    Args:
        features: Features from generated images [N, num_classes]
        splits: Number of splits for calculation
    
    Returns:
        tuple: (mean_is, std_is)
    """
    # Convert to probabilities if needed
    if features.max() > 1.0:
        features = F.softmax(torch.from_numpy(features), dim=1).numpy()
    
    # Calculate IS for each split
    n_samples = features.shape[0]
    split_size = n_samples // splits
    scores = []
    
    for i in range(splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < splits - 1 else n_samples
        
        split_features = features[start_idx:end_idx]
        
        # Calculate marginal and conditional entropy
        py = split_features.mean(axis=0)
        kl_div = split_features * (np.log(split_features + 1e-8) - np.log(py + 1e-8))
        kl_div = kl_div.sum(axis=1).mean()
        
        scores.append(np.exp(kl_div))
    
    return float(np.mean(scores)), float(np.std(scores))


def calculate_precision_recall(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Calculate precision and recall for multi-label classification.
    
    Args:
        predictions: Predicted probabilities [batch_size, num_classes]
        targets: True binary labels [batch_size, num_classes]
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary containing precision, recall, and F1 scores
    """
    pred_binary = (predictions > threshold).float()
    
    # Calculate per-class metrics
    tp = (pred_binary * targets).sum(dim=0)
    fp = (pred_binary * (1 - targets)).sum(dim=0)
    fn = ((1 - pred_binary) * targets).sum(dim=0)
    
    # Avoid division by zero
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Calculate macro averages
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()
    
    # Calculate micro averages
    micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8)
    micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision.item(),
        'micro_recall': micro_recall.item(),
        'micro_f1': micro_f1.item(),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
    }
