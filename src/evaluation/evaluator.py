"""
Diffusion model evaluator for inference and sample generation.
"""

import os
import time
import torch
import numpy as np
from typing import Optional, Dict, Any
from PIL import Image
import json
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ddpm import DDPM
from models.unet import UNet
from training.config import TrainingConfig
from training.utils import set_seed, get_device, load_checkpoint, setup_logging
from training.dataset import create_data_loaders, get_object_names
from .metrics import accuracy_score, calculate_precision_recall


class DiffusionEvaluator:
    """
    Evaluator class for diffusion model inference and evaluation.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Training/evaluation configuration
        """
        self.config = config
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup device
        self.device = get_device(config.device)
        
        # Setup logging
        self.logger = setup_logging(config.save_dir, 'INFO')
        self.logger.info(f"Starting evaluation: {config.run_name}")
        
        # Initialize models
        self._setup_models()
        
        # Setup data loaders for evaluation
        self._setup_data()
        
        # Load evaluator if provided
        if config.evaluator_ckpt:
            self._load_evaluator()
    
    def _setup_models(self):
        """Setup UNet and DDPM models."""
        self.logger.info("Setting up models...")
        
        # Get object names for num_classes
        object_names = get_object_names(self.config.data_dir)
        self.object_names = object_names
        num_classes = len(object_names)
        
        # Create UNet model
        self.unet = UNet(
            img_size=self.config.img_size,
            in_channels=3,
            out_channels=3,
            base_channels=self.config.base_channels,
            channel_multipliers=self.config.channel_multipliers,
            num_res_blocks=self.config.num_res_blocks,
            attention_resolutions=self.config.attention_resolutions,
            num_classes=num_classes,
            dropout=self.config.dropout,
            use_attention=self.config.use_attention,
        ).to(self.device)
        
        # Create DDPM model
        self.ddpm = DDPM(
            model=self.unet,
            timesteps=self.config.timesteps,
            beta_schedule=self.config.beta_schedule,
            loss_type=self.config.loss_type,
            use_classifier_guidance=self.config.use_classifier_guidance,
            classifier_guidance_scale=self.config.classifier_guidance_scale,
            classifier_guidance_start_step=self.config.classifier_guidance_start_step,
        ).to(self.device)
        
        # Load trained model
        if self.config.model_ckpt:
            self._load_trained_model()
        else:
            self.logger.warning("No model checkpoint provided!")
    
    def _setup_data(self):
        """Setup data loaders."""
        self.logger.info("Setting up data loaders...")
        
        self.data_loaders = create_data_loaders(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            img_size=(self.config.img_size, self.config.img_size),
            num_workers=self.config.num_workers,
            use_preset_size=self.config.use_preset_size
        )
        
        # Log dataset information
        for split, loader in self.data_loaders.items():
            size = len(loader.dataset)
            self.logger.info(f"{split.capitalize()} dataset size: {size}")
    
    def _load_trained_model(self):
        """Load trained diffusion model."""
        self.logger.info(f"Loading trained model from: {self.config.model_ckpt}")
        
        try:
            load_checkpoint(
                self.config.model_ckpt,
                self.unet,
                device=self.device,
                strict=False
            )
            self.logger.info("Trained model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading trained model: {e}")
            raise
    
    def _load_evaluator(self):
        """Load pretrained evaluator for accuracy calculation."""
        self.logger.info(f"Loading evaluator from: {self.config.evaluator_ckpt}")
        
        try:
            # Load evaluator checkpoint
            checkpoint = torch.load(self.config.evaluator_ckpt, map_location=self.device)
            
            # Create evaluator model (assuming it's a classifier)
            # You may need to adjust this based on your evaluator architecture
            from torchvision.models import resnet18
            self.evaluator = resnet18(num_classes=len(self.object_names))
            self.evaluator.load_state_dict(checkpoint)
            self.evaluator = self.evaluator.to(self.device)
            self.evaluator.eval()
            
            self.logger.info("Evaluator loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load evaluator: {e}")
            self.evaluator = None
    
    def generate_samples(
        self, 
        num_samples: Optional[int] = None,
        save_dir: Optional[str] = None,
        use_test_conditions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate samples using the trained diffusion model.
        
        Args:
            num_samples: Number of samples to generate (uses test set size if None)
            save_dir: Directory to save samples (uses config save_dir if None)
            use_test_conditions: Whether to use test set conditions
        
        Returns:
            dict: Generation results and statistics
        """
        self.logger.info("Starting sample generation...")
        self.unet.eval()
        
        if save_dir is None:
            save_dir = os.path.join(self.config.save_dir, 'generated_samples')
        os.makedirs(save_dir, exist_ok=True)
        
        results = {'samples': [], 'conditions': [], 'generation_time': 0}
        
        with torch.no_grad():
            if use_test_conditions and 'test' in self.data_loaders:
                # Use test set conditions
                test_loader = self.data_loaders['test']
                if num_samples is None:
                    num_samples = len(test_loader.dataset)
                
                start_time = time.time()
                
                for batch_idx, (_, test_labels) in enumerate(tqdm(test_loader, desc="Generating samples")):
                    if batch_idx * self.config.batch_size >= num_samples:
                        break
                    
                    batch_size = min(
                        self.config.batch_size,
                        num_samples - batch_idx * self.config.batch_size
                    )
                    labels = test_labels[:batch_size].to(self.device)
                    
                    # Generate samples
                    samples = self.ddpm.sample(
                        batch_size=batch_size,
                        class_labels=labels
                    )
                    
                    # Denormalize samples from [-1, 1] to [0, 1]
                    samples = (samples + 1) / 2
                    samples = torch.clamp(samples, 0, 1)
                    
                    # Save individual samples
                    for i in range(batch_size):
                        sample_idx = batch_idx * self.config.batch_size + i
                        
                        # Convert to PIL Image and save
                        sample_img = samples[i].cpu()
                        sample_pil = Image.fromarray(
                            (sample_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        )
                        
                        sample_path = os.path.join(save_dir, f'sample_{sample_idx:04d}.png')
                        sample_pil.save(sample_path)
                        
                        # Store results
                        results['samples'].append(sample_path)
                        
                        # Convert multi-hot labels to object names
                        label_indices = torch.where(labels[i] > 0.5)[0].cpu().numpy()
                        condition_objects = [self.object_names[idx] for idx in label_indices]
                        results['conditions'].append(condition_objects)
                
                results['generation_time'] = time.time() - start_time
                
            else:
                # Generate unconditional samples
                if num_samples is None:
                    num_samples = 32
                
                start_time = time.time()
                
                num_batches = (num_samples + self.config.batch_size - 1) // self.config.batch_size
                
                for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
                    batch_size = min(
                        self.config.batch_size,
                        num_samples - batch_idx * self.config.batch_size
                    )
                    
                    # Generate unconditional samples
                    samples = self.ddpm.sample(batch_size=batch_size)
                    
                    # Denormalize samples
                    samples = (samples + 1) / 2
                    samples = torch.clamp(samples, 0, 1)
                    
                    # Save individual samples
                    for i in range(batch_size):
                        sample_idx = batch_idx * self.config.batch_size + i
                        
                        sample_img = samples[i].cpu()
                        sample_pil = Image.fromarray(
                            (sample_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        )
                        
                        sample_path = os.path.join(save_dir, f'sample_{sample_idx:04d}.png')
                        sample_pil.save(sample_path)
                        
                        results['samples'].append(sample_path)
                        results['conditions'].append([])  # No conditions for unconditional
                
                results['generation_time'] = time.time() - start_time
        
        # Save generation metadata
        metadata = {
            'num_samples': len(results['samples']),
            'generation_time': results['generation_time'],
            'samples_per_second': len(results['samples']) / results['generation_time'],
            'config': vars(self.config),
            'conditions': results['conditions']
        }
        
        metadata_path = os.path.join(save_dir, 'generation_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(
            f"Generated {len(results['samples'])} samples in "
            f"{results['generation_time']:.2f}s "
            f"({len(results['samples']) / results['generation_time']:.2f} samples/s)"
        )
        
        return results
    
    def evaluate_samples(self, sample_dir: str) -> Dict[str, Any]:
        """
        Evaluate generated samples using the pretrained evaluator.
        
        Args:
            sample_dir: Directory containing generated samples
        
        Returns:
            dict: Evaluation results
        """
        if self.evaluator is None:
            self.logger.warning("No evaluator available for sample evaluation")
            return {}
        
        self.logger.info(f"Evaluating samples in: {sample_dir}")
        
        # Load samples and metadata
        metadata_path = os.path.join(sample_dir, 'generation_metadata.json')
        if not os.path.exists(metadata_path):
            self.logger.error("Generation metadata not found")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        conditions = metadata['conditions']
        sample_paths = [
            os.path.join(sample_dir, f'sample_{i:04d}.png')
            for i in range(metadata['num_samples'])
        ]
        
        # Evaluate samples
        predictions = []
        targets = []
        
        self.evaluator.eval()
        with torch.no_grad():
            for sample_path, condition in tqdm(
                zip(sample_paths, conditions), 
                desc="Evaluating samples",
                total=len(sample_paths)
            ):
                # Load and preprocess sample
                sample_img = Image.open(sample_path).convert('RGB')
                sample_tensor = torch.from_numpy(np.array(sample_img)).permute(2, 0, 1).float() / 255.0
                sample_tensor = sample_tensor.unsqueeze(0).to(self.device)
                
                # Get prediction
                pred = self.evaluator(sample_tensor)
                pred_probs = torch.sigmoid(pred)
                predictions.append(pred_probs.cpu())
                
                # Create target multi-hot vector
                target = torch.zeros(len(self.object_names))
                for obj_name in condition:
                    if obj_name in self.object_names:
                        obj_idx = self.object_names.index(obj_name)
                        target[obj_idx] = 1.0
                targets.append(target)
        
        # Calculate metrics
        predictions = torch.cat(predictions, dim=0)
        targets = torch.stack(targets, dim=0)
        
        accuracy = accuracy_score(predictions, targets)
        precision_recall = calculate_precision_recall(predictions, targets)
        
        evaluation_results = {
            'accuracy': accuracy,
            'num_samples': len(sample_paths),
            **precision_recall
        }
        
        # Save evaluation results
        eval_path = os.path.join(sample_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.logger.info(f"Evaluation results: Accuracy = {accuracy:.4f}")
        self.logger.info(f"Macro F1 = {precision_recall['macro_f1']:.4f}")
        
        return evaluation_results
    
    def run_inference(self) -> Dict[str, Any]:
        """
        Run complete inference pipeline: generation + evaluation.
        
        Returns:
            dict: Complete inference results
        """
        self.logger.info("Running complete inference pipeline...")
        
        # Generate samples
        generation_results = self.generate_samples()
        
        # Evaluate samples if evaluator is available
        if self.evaluator is not None:
            sample_dir = os.path.dirname(generation_results['samples'][0])
            evaluation_results = self.evaluate_samples(sample_dir)
            
            return {
                'generation': generation_results,
                'evaluation': evaluation_results
            }
        else:
            return {
                'generation': generation_results,
                'evaluation': {}
            }
