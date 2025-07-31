#!/bin/bash

# Demo script for inference with trained diffusion model
# This script demonstrates sample generation and evaluation

echo "=== Conditional Diffusion Model Inference Demo ==="

# Check if model checkpoint is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide a model checkpoint path"
    echo "Usage: $0 <model_checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  $0 ./results/my_experiment/checkpoints/best_model.pth"
    exit 1
fi

MODEL_CKPT="$1"

# Set default parameters
DATA_DIR=${DATA_DIR:-"./data/iclevr"}
SAVE_DIR=${SAVE_DIR:-"./results"}
RUN_NAME=${RUN_NAME:-"inference_$(date +%Y%m%d_%H%M%S)"}
NUM_SAMPLES=${NUM_SAMPLES:-32}

echo ""
echo "Configuration:"
echo "  Model checkpoint: $MODEL_CKPT"
echo "  Data directory: $DATA_DIR"
echo "  Save directory: $SAVE_DIR"
echo "  Run name: $RUN_NAME"
echo "  Number of samples: $NUM_SAMPLES"
echo ""

# Check if model checkpoint exists
if [ ! -f "$MODEL_CKPT" ]; then
    echo "❌ Error: Model checkpoint '$MODEL_CKPT' not found!"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory '$DATA_DIR' not found!"
    exit 1
fi

echo "✅ Model checkpoint and data directory found!"
echo ""

# Create results directory
mkdir -p "$SAVE_DIR"

echo "🎨 Generating samples..."
echo ""

# Generate samples
python -m src.main \
    --mode sample \
    --run_name "$RUN_NAME" \
    --model_ckpt "$MODEL_CKPT" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --batch_size 16 \
    --img_size 64 \
    --timesteps 400 \
    --sampling_timesteps 50 \
    --beta_schedule cosine \
    --use_attention \
    --seed 42

SAMPLE_DIR="$SAVE_DIR/generated_samples"

echo ""
echo "✅ Sample generation completed!"
echo ""
echo "Generated samples saved to: $SAMPLE_DIR"
echo ""

# Check if evaluator checkpoint is available for evaluation
if [ -n "$EVALUATOR_CKPT" ] && [ -f "$EVALUATOR_CKPT" ]; then
    echo "📊 Running evaluation with provided evaluator..."
    echo ""
    
    python -m src.main \
        --mode inference \
        --run_name "${RUN_NAME}_eval" \
        --model_ckpt "$MODEL_CKPT" \
        --evaluator_ckpt "$EVALUATOR_CKPT" \
        --data_dir "$DATA_DIR" \
        --save_dir "$SAVE_DIR" \
        --batch_size 16 \
        --img_size 64 \
        --timesteps 400 \
        --sampling_timesteps 50 \
        --beta_schedule cosine \
        --use_attention \
        --seed 42
    
    echo "📈 Evaluation results saved to: $SAVE_DIR/${RUN_NAME}_eval/"
else
    echo "ℹ️  No evaluator checkpoint provided. Skipping evaluation."
    echo "   To run evaluation, set EVALUATOR_CKPT environment variable:"
    echo "   export EVALUATOR_CKPT=/path/to/evaluator.pth"
    echo "   $0 $MODEL_CKPT"
fi

echo ""
echo "📁 Results summary:"
echo "  🖼️  Generated samples: $SAMPLE_DIR/"
echo "  📋 Generation metadata: $SAMPLE_DIR/generation_metadata.json"

if [ -f "$SAMPLE_DIR/evaluation_results.json" ]; then
    echo "  📊 Evaluation results: $SAMPLE_DIR/evaluation_results.json"
    
    # Display key metrics if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        echo "📈 Key evaluation metrics:"
        echo "  Accuracy: $(jq -r '.accuracy' $SAMPLE_DIR/evaluation_results.json)"
        echo "  Macro F1: $(jq -r '.macro_f1' $SAMPLE_DIR/evaluation_results.json)"
        echo "  Macro Precision: $(jq -r '.macro_precision' $SAMPLE_DIR/evaluation_results.json)"
        echo "  Macro Recall: $(jq -r '.macro_recall' $SAMPLE_DIR/evaluation_results.json)"
    fi
fi

echo ""
echo "✨ Inference demo completed successfully!"
