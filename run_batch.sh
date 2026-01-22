#!/bin/bash
# Batch processing script for OMGs multi-model evaluation
# This script runs EHR structuring and multiple model configurations, then extracts results

set -e  # Exit on error

# Configuration
INPUT_FILE="./input_ehr/test_batch.jsonl"
OUTPUT_EHR="./output_ehr/test_batch.jsonl"
TXT_DIR="./output_ehr/txt_out"
PROMPTS_FILE="./config/prompts.json"

# Array to store output_answer directories
declare -a OUTPUT_DIRS=()

echo "=========================================="
echo "OMGs Batch Processing Script"
echo "=========================================="

# Step 1: Run EHR Structurer
echo ""
echo "[STEP 1] Running EHR Structurer..."
python ehr_structurer.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_EHR" \
  --deployment gpt-5-mini \
  --prompts "$PROMPTS_FILE" \
  --txt-dir "$TXT_DIR" \
  --provider azure

if [ ! -f "$OUTPUT_EHR" ]; then
    echo "[ERROR] EHR structuring failed. Output file not found: $OUTPUT_EHR"
    exit 1
fi

echo "[SUCCESS] EHR structuring completed: $OUTPUT_EHR"

# Step 2: Run all model configurations
echo ""
echo "[STEP 2] Running model configurations..."

# Record start time to identify directories created during this run
SCRIPT_START_TIME=$(date +%s)

# Function to run main.py and capture output directory
run_main_py() {
    local agent=$1
    local provider=$2
    local model=$3
    local description=$4
    
    echo ""
    echo "----------------------------------------"
    echo "Running: $description"
    echo "Agent: $agent, Provider: $provider, Model: $model"
    echo "----------------------------------------"
    
    # Record timestamp before running
    local before_time=$(date +%s)
    
    # Run the command
    python main.py \
        --input_path "$OUTPUT_EHR" \
        --agent "$agent" \
        --provider "$provider" \
        --model "$model" \
        --num_samples 1
    
    # Find the most recent output_answer directory for this agent created after before_time
    # Pattern: output_answer/{agent}_YYYY-MM-DD_HH-MM-SS
    local latest_dir=""
    if [ -d "output_answer" ]; then
        for dir in output_answer/${agent}_*; do
            if [ -d "$dir" ]; then
                local dir_time=$(stat -f %m "$dir" 2>/dev/null || stat -c %Y "$dir" 2>/dev/null || echo 0)
                if [ "$dir_time" -ge "$before_time" ]; then
                    latest_dir="$dir"
                fi
            fi
        done
    fi
    
    if [ -n "$latest_dir" ] && [ -d "$latest_dir" ]; then
        OUTPUT_DIRS+=("$latest_dir")
        echo "[INFO] Output directory recorded: $latest_dir"
    else
        # Fallback: just get the most recent one
        local fallback_dir=$(ls -td output_answer/${agent}_* 2>/dev/null | head -1)
        if [ -n "$fallback_dir" ] && [ -d "$fallback_dir" ]; then
            OUTPUT_DIRS+=("$fallback_dir")
            echo "[INFO] Output directory recorded (fallback): $fallback_dir"
        else
            echo "[WARNING] Could not find output directory for $agent"
        fi
    fi
}

# Chair-SA baseline group
run_main_py "chair_sa" "azure" "gpt-5.1" "Chair-SA (Baseline)"
run_main_py "chair_sa_k" "azure" "gpt-5.1" "Chair-SA(K) - Knowledge only"
run_main_py "chair_sa_kep" "azure" "gpt-5.1" "Chair-SA(K+EP) - Knowledge + Evidence Pack"

# OMGs baseline group
run_main_py "omgs" "azure" "gpt-5.1" "OMGs Baseline - GPT-5.1 (Azure)"

# OpenRouter comparison group
run_main_py "omgs" "openrouter" "anthropic/claude-opus-4.5" "OMGs - Claude Opus 4.5"
run_main_py "omgs" "openrouter" "deepseek/deepseek-v3.2" "OMGs - DeepSeek V3.2"
run_main_py "omgs" "openrouter" "deepseek/deepseek-r1" "OMGs - DeepSeek R1"
run_main_py "omgs" "openrouter" "google/gemini-3-pro-preview" "OMGs - Gemini 3 Pro Preview"
run_main_py "omgs" "openrouter" "meta-llama/llama-3.1-405b-instruct" "OMGs - Llama 3.1 405B Instruct"
run_main_py "omgs" "openrouter" "qwen/qwen3-235b-a22b-2507" "OMGs - Qwen3 235B"

echo ""
echo "[STEP 2] All model configurations completed."
echo "[INFO] Total output directories: ${#OUTPUT_DIRS[@]}"

# Step 3: Extract and organize results
echo ""
echo "[STEP 3] Extracting and organizing results..."

# Create output_batch directory
OUTPUT_BATCH_DIR="./output_batch"
mkdir -p "$OUTPUT_BATCH_DIR"

# Run Python script to extract results
# Pass OUTPUT_DIRS array to Python script via file
OUTPUT_DIRS_FILE=$(mktemp)
printf '%s\n' "${OUTPUT_DIRS[@]}" > "$OUTPUT_DIRS_FILE"
export OUTPUT_DIRS_FILE

python3 << 'PYTHON_SCRIPT'
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Read output directories from file
output_dirs_file = os.environ.get('OUTPUT_DIRS_FILE', '')
output_dirs = []

if output_dirs_file and os.path.exists(output_dirs_file):
    with open(output_dirs_file, 'r') as f:
        output_dirs = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]
    os.remove(output_dirs_file)

# If no directories from file, fallback to scanning
if not output_dirs:
    if os.path.exists("output_answer") and os.path.isdir("output_answer"):
        for subdir in os.listdir("output_answer"):
            subdir_path = os.path.join("output_answer", subdir)
            if os.path.isdir(subdir_path):
                results_json = os.path.join(subdir_path, "results.json")
                if os.path.exists(results_json):
                    output_dirs.append(subdir_path)

print(f"[INFO] Found {len(output_dirs)} output directories")

# Group results by meta_info
patient_results = defaultdict(list)

for output_dir in output_dirs:
    results_json = os.path.join(output_dir, "results.json")
    if not os.path.exists(results_json):
        print(f"[WARNING] results.json not found in {output_dir}")
        continue
    
    try:
        with open(results_json, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Process each result in the array
        for result in results:
            meta_info = result.get('meta_info')
            if not meta_info:
                print(f"[WARNING] Missing meta_info in result from {output_dir}")
                continue
            
            # Get mode and model
            mode = result.get('mode', result.get('agent_mode', 'unknown'))
            model = result.get('model', 'unknown')
            
            # Create entry
            entry = {
                'mode': mode,
                'model': model,
                'response': result.get('response', ''),
                'scene': result.get('scene'),
                'error': result.get('error'),
            }
            
            patient_results[meta_info].append(entry)
    
    except Exception as e:
        print(f"[ERROR] Failed to process {results_json}: {e}")

# Write results for each patient
output_batch_dir = "./output_batch"
os.makedirs(output_batch_dir, exist_ok=True)

for meta_info, entries in patient_results.items():
    # Create patient directory
    patient_dir = os.path.join(output_batch_dir, meta_info)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Write results.txt
    output_file = os.path.join(patient_dir, "results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            mode = entry['mode']
            model = entry['model']
            response = entry['response']
            error = entry.get('error')
            
            # Write header
            f.write("=" * 50 + "\n")
            f.write(f"{mode}_{model}\n")
            f.write("=" * 50 + "\n")
            
            # Write content
            if error:
                f.write(f"[ERROR] {error}\n")
            elif response:
                f.write(f"{response}\n")
            else:
                f.write("[No response]\n")
            
            f.write("\n")
    
    print(f"[INFO] Wrote results for patient: {meta_info} ({len(entries)} entries)")

print(f"[SUCCESS] Results extracted to {output_batch_dir}")
PYTHON_SCRIPT

# Clean up temp file if it still exists
rm -f "$OUTPUT_DIRS_FILE" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Batch processing completed!"
echo "=========================================="
echo "Results are organized in: ./output_batch/{meta_info}/results.txt"
echo ""
