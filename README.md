# Conditional Diffusion Model for i-CLEVR Generation

A PyTorch implementation of conditional Denoising Diffusion Probabilistic Models (DDPM) for generating i-CLEVR dataset images based on object specifications.

## 🎯 Features

- **Conditional Generation**: Generate images based on object specifications (shapes, colors)
- **Advanced UNet Architecture**: Custom UNet with attention mechanisms and residual blocks
- **Flexible Training**: Support for different beta schedules, loss functions, and optimization strategies
- **Comprehensive Evaluation**: Built-in metrics for assessing generation quality
- **Memory Efficient**: Gradient accumulation and memory optimization options
- **Experiment Tracking**: Integration with Weights & Biases for monitoring training progress

## 🏗️ Architecture

### Model Components

- **UNet with Attention**: Custom UNet architecture with self-attention layers for improved long-range dependencies
- **Conditional DDPM**: Diffusion model that supports conditioning on multiple object labels
- **Multi-Object Support**: Handle images with 1-3 objects using multi-hot encoding
- **Classifier Guidance**: Optional classifier-guided sampling for improved label accuracy

### Key Features

- **Residual Blocks**: Enhanced with time embedding and up/down sampling
- **Self-Attention**: Captures long-range spatial dependencies
- **Time Embedding**: Sinusoidal embedding for diffusion timesteps
- **Class Embedding**: Multi-hot encoding for conditional generation

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd diffusion-model

# Install dependencies
pip install -r requirements.txt

# Prepare your data directory structure:
# data/
# ├── iclevr/
# │   ├── train.json
# │   ├── test.json
# │   ├── new_test.json
# │   ├── objects.json
# │   └── images/
```

## 🚀 Quick Start

### Training

```bash
# Basic training
python -m src.main --mode train --run_name my_experiment

# Training with custom parameters
python -m src.main \
    --mode train \
    --run_name ddpm_experiment \
    --data_dir ./data/iclevr \
    --batch_size 32 \
    --epochs 200 \
    --lr 2e-4 \
    --timesteps 400 \
    --beta_schedule cosine \
    --use_attention \
    --use_wandb
```

### Inference

```bash
# Generate samples using trained model
python -m src.main \
    --mode sample \
    --model_ckpt ./results/my_experiment/checkpoints/best_model.pth \
    --data_dir ./data/iclevr

# Run complete evaluation pipeline
python -m src.main \
    --mode inference \
    --model_ckpt ./results/my_experiment/checkpoints/best_model.pth \
    --evaluator_ckpt ./pretrained/evaluator.pth \
    --data_dir ./data/iclevr
```

## 📊 Dataset Format

### i-CLEVR Dataset Structure

```
data/iclevr/
├── train.json          # Training labels: {"filename": ["object1", "object2", ...]}
├── test.json           # Test conditions: [["object1"], ["object1", "object2"], ...]
├── new_test.json       # Additional test conditions
├── objects.json        # Object definitions: {"object_name": index}
└── images/             # Training images directory
```

### Object Specifications

- **Shapes**: cube, sphere, cylinder
- **Colors**: red, green, blue, yellow, cyan, purple, brown, gray
- **Total Objects**: 24 (3 shapes × 8 colors)
- **Objects per Image**: 1-3 objects

## ⚙️ Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 64 | Training batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--timesteps` | 400 | Number of diffusion timesteps |
| `--sampling_timesteps` | 50 | Sampling steps (DDIM acceleration) |
| `--beta_schedule` | cosine | Beta schedule ('linear' or 'cosine') |
| `--use_attention` | False | Enable attention layers |

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_channels` | 128 | Base number of UNet channels |
| `--num_res_blocks` | 2 | Residual blocks per level |
| `--channel_multipliers` | [1,2,4,8] | Channel scaling factors |
| `--attention_resolutions` | [16,8] | Resolutions for attention |

### Memory Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gradient_accumulation_steps` | 1 | Gradient accumulation |
| `--reduce_memory` | False | Enable memory optimization |
| `--grad_clip` | 1.0 | Gradient clipping threshold |

## 📈 Monitoring and Logging

### Local Logging

- Training logs: `results/{run_name}/logs/training.log`
- Checkpoints: `results/{run_name}/checkpoints/`
- Generated samples: `results/{run_name}/samples/`
- Training curves: `results/{run_name}/training_curves.png`

### Weights & Biases Integration

```bash
# Enable W&B logging
python -m src.main \
    --mode train \
    --use_wandb \
    --wandb_project "diffusion-iclevr" \
    --wandb_name "experiment-1"
```

For comprehensive experimental analysis, preprocessing optimization results, model architecture comparisons, and detailed technical insights, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## 🎯 Performance Tips

### Training Optimization

1. **Batch Size**: Use largest batch size that fits in memory (32-128)
2. **Learning Rate**: Start with 2e-4, adjust based on convergence
3. **Beta Schedule**: Cosine generally works better than linear
4. **Attention**: Enable for better quality but slower training
5. **Gradient Accumulation**: Use when memory is limited

### Memory Management

```bash
# For limited GPU memory
python -m src.main \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --reduce_memory \
    --num_workers 2
```

### Quality vs Speed Trade-offs

```bash
# High quality (slower)
--timesteps 1000 --sampling_timesteps 250 --use_attention

# Balanced (recommended)
--timesteps 400 --sampling_timesteps 50 --use_attention

# Fast (lower quality)
--timesteps 200 --sampling_timesteps 25
```

## 📁 Project Structure

```
diffusion-model/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ddpm.py          # DDPM implementation
│   │   ├── unet.py          # UNet architecture
│   │   └── layers.py        # Neural network layers
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Main training loop
│   │   ├── config.py        # Configuration management
│   │   ├── dataset.py       # Data loading utilities
│   │   └── utils.py         # Training utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py     # Model evaluation
│   │   └── metrics.py       # Evaluation metrics
│   ├── evaluator.py         # Pre-trained ResNet18 evaluator
│   ├── preprocess_images.py # Image preprocessing utilities
│   └── main.py              # Entry point
├── demo/
│   ├── train.sh             # Training demo script
│   ├── inference.sh         # Inference demo script
│   └── quickstart.py        # Python demo
├── data/                    # Dataset directory
├── results/                 # Training outputs
├── requirements.txt         # Dependencies
├── TECHNICAL_REPORT.md      # Detailed technical implementation report
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Reduce batch size and enable gradient accumulation
   --batch_size 16 --gradient_accumulation_steps 4
   ```

2. **Slow Training**

   ```bash
   # Disable attention or reduce model size
   --base_channels 64 --num_res_blocks 1
   ```

3. **Poor Generation Quality**

   ```bash
   # Increase model capacity and training time
   --base_channels 256 --epochs 300 --use_attention
   ```

### Performance Monitoring

- Monitor training loss for convergence
- Check generated samples during training
- Use W&B for real-time monitoring
- Validate with evaluation metrics

## 📚 References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📬 Contact

For questions and support, please open an issue on the repository.
