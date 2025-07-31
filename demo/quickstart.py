"""
Quick start demo for conditional diffusion model.

This script demonstrates how to use the diffusion model API programmatically.
"""

import os
import torch
from src.models import DDPM, UNet
from src.training import TrainingConfig, DiffusionTrainer
from src.evaluation import DiffusionEvaluator


def demo_training():
    """Demonstrate model training with minimal configuration."""
    print("🚀 Demo: Training Conditional Diffusion Model")
    print("=" * 50)
    
    # Create minimal training configuration
    config = TrainingConfig(
        mode='train',
        run_name='demo_quickstart',
        data_dir='./data/iclevr',
        save_dir='./results',
        batch_size=16,
        epochs=5,  # Short demo
        lr=2e-4,
        img_size=64,
        timesteps=200,  # Reduced for faster demo
        sampling_timesteps=25,
        beta_schedule='cosine',
        use_attention=False,  # Disabled for faster training
        checkpoint_freq=2,
        sample_freq=2,
        seed=42
    )
    
    print(f"Configuration: {config.run_name}")
    print(f"Data directory: {config.data_dir}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(config.data_dir):
        print(f"❌ Error: Data directory '{config.data_dir}' not found!")
        print("Please prepare your dataset first.")
        return
    
    try:
        # Initialize trainer
        print("🔧 Initializing trainer...")
        trainer = DiffusionTrainer(config)
        
        # Start training
        print("🎯 Starting training...")
        trainer.train()
        
        print("✅ Training completed successfully!")
        print(f"Results saved to: {trainer.run_dir}")
        
        return trainer.run_dir
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


def demo_inference(model_checkpoint_path):
    """Demonstrate model inference with a trained checkpoint."""
    print("\n🎨 Demo: Generating Samples with Trained Model")
    print("=" * 50)
    
    if not os.path.exists(model_checkpoint_path):
        print(f"❌ Error: Model checkpoint '{model_checkpoint_path}' not found!")
        return
    
    # Create inference configuration
    config = TrainingConfig(
        mode='sample',
        run_name='demo_inference',
        model_ckpt=model_checkpoint_path,
        data_dir='./data/iclevr',
        save_dir='./results',
        batch_size=8,
        img_size=64,
        timesteps=200,
        sampling_timesteps=25,
        beta_schedule='cosine',
        use_attention=False,
        seed=42
    )
    
    print(f"Model checkpoint: {config.model_ckpt}")
    print(f"Generation mode: {config.mode}")
    print()
    
    try:
        # Initialize evaluator
        print("🔧 Initializing evaluator...")
        evaluator = DiffusionEvaluator(config)
        
        # Generate samples
        print("🎨 Generating samples...")
        results = evaluator.generate_samples(num_samples=16)
        
        print("✅ Sample generation completed!")
        print(f"Generated {len(results['samples'])} samples")
        print(f"Generation time: {results['generation_time']:.2f} seconds")
        print(f"Speed: {len(results['samples']) / results['generation_time']:.2f} samples/sec")
        
        # Display sample paths
        print("\n📁 Generated samples:")
        for i, sample_path in enumerate(results['samples'][:5]):  # Show first 5
            print(f"  {i+1}. {sample_path}")
        
        if len(results['samples']) > 5:
            print(f"  ... and {len(results['samples']) - 5} more")
        
        return results
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None


def demo_model_creation():
    """Demonstrate manual model creation and configuration."""
    print("\n🏗️  Demo: Manual Model Creation")
    print("=" * 50)
    
    try:
        # Model parameters
        img_size = 64
        num_classes = 24  # i-CLEVR has 24 objects (3 shapes × 8 colors)
        
        print(f"Creating UNet model...")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Number of classes: {num_classes}")
        
        # Create UNet model
        unet = UNet(
            img_size=img_size,
            in_channels=3,
            out_channels=3,
            base_channels=64,  # Smaller for demo
            channel_multipliers=[1, 2, 4],  # Simpler architecture
            num_res_blocks=1,
            attention_resolutions=[16],
            num_classes=num_classes,
            dropout=0.1,
            use_attention=True,
        )
        
        print(f"✅ UNet created successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in unet.parameters())
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Create DDPM wrapper
        print(f"\nCreating DDPM model...")
        ddpm = DDPM(
            model=unet,
            timesteps=200,
            beta_schedule='cosine',
            loss_type='l1',
            use_classifier_guidance=False,
        )
        
        print(f"✅ DDPM created successfully!")
        print(f"  Timesteps: {ddpm.timesteps}")
        print(f"  Beta schedule: cosine")
        print(f"  Loss type: L1")
        
        # Test forward pass
        print(f"\n🧪 Testing forward pass...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ddpm = ddpm.to(device)
        
        # Create dummy batch
        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
        dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).float().to(device)
        
        print(f"  Input shape: {dummy_images.shape}")
        print(f"  Labels shape: {dummy_labels.shape}")
        print(f"  Device: {device}")
        
        # Forward pass
        with torch.no_grad():
            loss = ddpm(dummy_images, dummy_labels)
        
        print(f"✅ Forward pass successful!")
        print(f"  Loss: {loss.item():.6f}")
        
        return ddpm
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def main():
    """Run all demos."""
    print("🎉 Conditional Diffusion Model - Quick Start Demo")
    print("=" * 60)
    print()
    
    # Demo 1: Manual model creation
    model = demo_model_creation()
    
    if model is not None:
        print("\n" + "=" * 60)
        
        # Demo 2: Training (optional, comment out for quick demo)
        print("Skipping training demo (takes time)...")
        print("To run training demo, uncomment the training section in main()")
        
        # Uncomment below for training demo:
        # run_dir = demo_training()
        # 
        # if run_dir:
        #     # Demo 3: Inference with trained model
        #     best_model_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
        #     if os.path.exists(best_model_path):
        #         demo_inference(best_model_path)
        #     else:
        #         print("No trained model found for inference demo")
    
    print("\n✨ Demo completed!")
    print("\nNext steps:")
    print("  1. Prepare your i-CLEVR dataset in ./data/iclevr/")
    print("  2. Run training: ./demo/train.sh")
    print("  3. Generate samples: ./demo/inference.sh <checkpoint_path>")
    print("  4. Explore the codebase in src/ directory")


if __name__ == '__main__':
    main()
