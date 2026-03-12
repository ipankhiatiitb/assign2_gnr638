#!/bin/bash

# GNR 638 Assignment 2: Complete Few-Shot Experiment Pipeline
# Run all experiments across 100%, 20%, and 5% data regimes

set -e  # Exit on error

echo "=========================================="
echo "GNR 638 Assignment 2: Full Pipeline"
echo "All Few-Shot Settings (100%, 20%, 5%)"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

# PHASE 0: Create all few-shot splits
echo "PHASE 0: Creating Few-Shot Splits"
echo "===================================="
python create_splits.py
echo ""

# Configuration
models=("resnet50" "efficientnetb0" "inceptionv3")
modes=("linear_probe" "last_block_finetune" "partial_finetune" "selective_20percent_last" "selective_20percent_random" "full_finetune")
fewshots=(100 20 5)

# PHASE 1 & 2: Training Loop
echo "PHASES 1-2: Training All Configurations"
echo "========================================"

for pct in "${fewshots[@]}"; do
    if [ $pct -eq 100 ]; then
        max_epochs=30
        echo ""
        echo "========== $pct% Few-Shot (Max $max_epochs epochs) =========="
    else
        max_epochs=20
        echo ""
        echo "========== $pct% Few-Shot (Max $max_epochs epochs) =========="
    fi
    
    for model in "${models[@]}"; do
        echo ""
        echo "--- $model ($pct%, $max_epochs epochs) ---"
        
        for mode in "${modes[@]}"; do
            echo "  ► $mode"
            python train.py \
                --model $model \
                --training-mode $mode \
                --few-shot-percentage $pct \
                --epochs $max_epochs \
                --batch-size 32
            echo ""
        done
    done
done

# PHASE 3: Advanced Evaluation
echo ""
echo "PHASE 3: Advanced Evaluation (Corruption + Layer-Wise)"
echo "======================================================"

for pct in "${fewshots[@]}"; do
    echo ""
    echo "========== Evaluating $pct% Models =========="
    
    for model in "${models[@]}"; do
        echo "  ► Evaluating $model ($pct%)"
        python evaluate_advanced.py \
            --model $model \
            --model-path "results/${model}_${pct}pct_final_model.pth" \
            --batch-size 32
        echo ""
    done
done

echo ""
echo "=========================================="
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "End Time: $(date)"
echo ""
echo "Output files in: results/"
echo "Total configurations: 54 training + 9 evaluation"
echo "=========================================="
