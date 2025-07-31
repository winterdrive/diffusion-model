"""
Pretrained classifier for evaluating generated i-CLEVR images.

This module provides a ResNet18-based classifier for computing classification accuracy
on generated images from the diffusion model.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os


class EvaluationModel:
    """
    Evaluation model for i-CLEVR image classification.
    
    Based on ResNet18 with modified final layer for 24-class multi-label classification.
    The model is trained on i-CLEVR dataset with 1-5 objects and 64x64 resolution.
    
    Usage:
        evaluator = EvaluationModel(checkpoint_path='./checkpoint.pth')
        accuracy = evaluator.eval(images, labels)
    
    Args:
        images (torch.Tensor): Batch of images with shape (batch_size, 3, 64, 64)
        labels (torch.Tensor): One-hot labels with shape (batch_size, 24)
    
    Note:
        Images should be normalized with transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    """
    
    def __init__(self, checkpoint_path=None, device=None):
        """
        Initialize evaluation model.
        
        Args:
            checkpoint_path (str): Path to pretrained checkpoint file
            device (torch.device): Device to run model on
        """
        if checkpoint_path is None:
            checkpoint_path = './checkpoint.pth'
            
        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        
        self.device = device
        print(f"Loading evaluation model from: {checkpoint_path}")
        print(f"Using device: {device}")
        
        # Check if checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create ResNet18 model
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 24),
            nn.Sigmoid()
        )
        
        # Load pretrained weights
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.to(device)
        self.resnet18.eval()
        
        self.num_classes = 24
        
    def compute_accuracy(self, predictions, onehot_labels):
        """
        Compute top-k accuracy for multi-label classification.
        
        For each sample, we take the top-k predictions where k is the number of 
        active labels in the ground truth, then compute accuracy.
        
        Args:
            predictions (torch.Tensor): Model predictions of shape (batch_size, num_classes)
            onehot_labels (torch.Tensor): One-hot ground truth labels of shape (batch_size, num_classes)
        
        Returns:
            float: Accuracy score
        """
        batch_size = predictions.size(0)
        correct = 0
        total = 0
        
        for i in range(batch_size):
            # Number of active labels for this sample
            k = int(onehot_labels[i].sum().item())
            total += k
            
            # Get top-k predictions and ground truth indices
            pred_values, pred_indices = predictions[i].topk(k)
            true_values, true_indices = onehot_labels[i].topk(k)
            
            # Count correct predictions
            for pred_idx in pred_indices:
                if pred_idx in true_indices:
                    correct += 1
                    
        return correct / total if total > 0 else 0.0
        
    def eval(self, images, labels):
        """
        Evaluate accuracy on a batch of images and labels.
        
        Args:
            images (torch.Tensor): Batch of images with shape (batch_size, 3, 64, 64)
            labels (torch.Tensor): One-hot labels with shape (batch_size, 24)
        
        Returns:
            float: Classification accuracy
        """
        with torch.no_grad():
            # Ensure inputs are on correct device
            if images.device != self.device:
                images = images.to(self.device)
            if labels.device != self.device:
                labels = labels.to(self.device)
                
            # Forward pass
            predictions = self.resnet18(images)
            
            # Compute accuracy (move to CPU for computation)
            accuracy = self.compute_accuracy(predictions.cpu(), labels.cpu())
            
            return accuracy
    
    def __call__(self, images, labels):
        """Make the model callable."""
        return self.eval(images, labels)


# Legacy class name for backward compatibility
evaluation_model = EvaluationModel
