#!/bin/bash

# Demo script for training conditional diffusion model
# This script demonstrates a complete training workflow

echo "=== Conditional Diffusion Model Training Demo ==="
echo "This script will train a diffusion model on the i-CLEVR dataset"

# Set default parameters
DATA_DIR=${DATA_DIR:-"./data/iclevr"}
SAVE_DIR=${SAVE_DIR:-"./results"}
RUN_NAME=${RUN_NAME:-"demo_training_$(date +%Y%m%d_%H%M%S)"}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Save directory: $SAVE_DIR"
echo "  Run name: $RUN_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory '$DATA_DIR' not found!"
    echo "Please prepare your dataset with the following structure:"
    echo "  $DATA_DIR/"
    echo "  ├── train.json"
    echo "  ├── test.json"
    echo "  ├── new_test.json"
    echo "  ├── objects.json"
    echo "  └── images/"
    exit 1
fi

# Check required files
REQUIRED_FILES=("train.json" "test.json" "objects.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "❌ Error: Required file '$DATA_DIR/$file' not found!"
        exit 1
    fi
done

echo "✅ Data directory structure looks good!"
echo ""

# Create results directory
mkdir -p "$SAVE_DIR"

echo "🚀 Starting training..."
echo ""

# Train the model
python -m src.main \
    --mode train \
    --run_name "$RUN_NAME" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr 2e-4 \
    --min_lr_factor 0.05 \
    --img_size 64 \
    --timesteps 400 \
    --sampling_timesteps 50 \
    --beta_schedule cosine \
    --loss_type l1 \
    --base_channels 128 \
    --num_res_blocks 2 \
    --use_attention \
    --dropout 0.1 \
    --grad_clip 1.0 \
    --checkpoint_freq 10 \
    --sample_freq 5 \
    --gradient_accumulation_steps 1 \
    --seed 42

echo ""
echo "✅ Training completed!"
echo ""
echo "Results saved to: $SAVE_DIR/$RUN_NAME"
echo ""
echo "Files generated:"
echo "  📊 Training logs: $SAVE_DIR/$RUN_NAME/logs/"
echo "  💾 Model checkpoints: $SAVE_DIR/$RUN_NAME/checkpoints/"
echo "  🖼️  Generated samples: $SAVE_DIR/$RUN_NAME/samples/"
echo "  📈 Training curves: $SAVE_DIR/$RUN_NAME/training_curves.png"
echo ""
echo "To generate samples with your trained model:"
echo "  ./demo/inference.sh $SAVE_DIR/$RUN_NAME/checkpoints/best_model.pth"
