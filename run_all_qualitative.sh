#!/bin/bash

# Configuration
REPORT_DIR="reports"
BASE_MODEL="output/real_full_kd"
TEACHER="EleutherAI/pythia-1.4b"

mkdir -p "$REPORT_DIR"

export PYTHONPATH=$PYTHONPATH:.

# Explicit list of experiments to evaluate
EXPERIMENTS=(
    "real_full_kd"
    "real_topk_k4"
    "real_topk_k8"
    "real_topk_k16"
    "real_wikitext_sampling_k50"
)

echo "Starting qualitative analysis on selected experiments..."

for dir_name in "${EXPERIMENTS[@]}"; do
    dir_path="output/$dir_name"
    
    if [ ! -d "$dir_path" ]; then
        echo "Warning: Directory $dir_path not found. Skipping."
        continue
    fi

    echo "----------------------------------------------------"
    echo "Processing experiment: $dir_name"
    
    # Run the analysis
    # We compare against real_full_kd as a reference, unless the experiment is real_full_kd itself.
    if [ "$dir_name" == "real_full_kd" ]; then
        # For the baseline, we just run it alone (it will be compared against the Teacher)
        python scripts/qualitative_analysis.py \
            --teacher_path "$TEACHER" \
            --student_paths "$dir_path" \
            --student_names "Full_KD" \
            --output_path "$REPORT_DIR/${dir_name}_report.md" \
            --num_val_samples 50
    else
        # For others, compare against the Full KD baseline
        python scripts/qualitative_analysis.py \
            --teacher_path "$TEACHER" \
            --student_paths "$BASE_MODEL" "$dir_path" \
            --student_names "Full_KD" "$dir_name" \
            --output_path "$REPORT_DIR/${dir_name}_report.md" \
            --num_val_samples 50
    fi

done

echo "----------------------------------------------------"
echo "Qualitative analysis complete. Reports stored in $REPORT_DIR/"
ls -lh "$REPORT_DIR"
