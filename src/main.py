"""
Main entry point for diffusion model training and inference.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.training import DiffusionTrainer, get_args


def main():
    """Main function for training and inference."""
    # Parse configuration
    config = get_args()
    
    print(f"Starting diffusion model {config.mode}...")
    print(f"Configuration: {config.run_name}")
    
    if config.mode == 'train':
        # Training mode
        trainer = DiffusionTrainer(config)
        trainer.train()
        
    elif config.mode == 'inference':
        # Inference mode
        from src.evaluation import DiffusionEvaluator
        evaluator = DiffusionEvaluator(config)
        evaluator.run_inference()
        
    elif config.mode == 'sample':
        # Sampling mode
        from src.evaluation import DiffusionEvaluator  
        evaluator = DiffusionEvaluator(config)
        evaluator.generate_samples()
        
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == '__main__':
    main()
