#!/bin/bash
#SBATCH --job-name=bias_exp4
#SBATCH --output=/home/fast/trabelb1/projects/exp4_mistral/exp4%j.out
#SBATCH --error=/home/fast/trabelb1/projects/exp4_mistral/exp4%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16

source ~/.bash_profile
REPO_LOCATION="/home/fast/trabelb1/projects/exp4_mistral"
nvidia-smi
cd $REPO_LOCATION
export PYTHONPATH=$(pwd)

# Helper function to run and notify
run_eval() {
    local model=$1
    echo "--------------------------------------------------"
    echo "RUNNING: $model"
    echo "--------------------------------------------------"
    
    # Create a temp file to capture output for Telegram
    tmp_log=$(mktemp)
    
    # Run command: removed --quiet, added tee to keep slurm logs intact
    uv run exp4-internal_repr_transformers.py --model "$model" --outdir data --trust-remote-code 2>&1 | tee "$tmp_log"
    
    # Prepare message: take last 20 lines (summary), escape backticks
    raw_summary=$(tail -n 15 "$tmp_log")
    clean_summary="${raw_summary//\`/’}"
    
    send_telegram "✅ Finished: $model\n\`\`\`\n$clean_summary\n\`\`\`"
    
    rm "$tmp_log"
}

# 1. Reasoning Models (OpenAI & Qwen)
run_eval "openai/gpt-oss-20b"
run_eval "openai/gpt-oss-120b"
run_eval "Qwen/Qwen3-32B"
run_eval "Qwen/Qwen3-Next-80B-A3B-Instruct"
run_eval "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
run_eval "dphn/dolphin-2.9.1-yi-1.5-34b"
run_eval "01-ai/Yi-1.5-34B-Chat"
# 2. Distilled Reasoning
run_eval "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# 3. Meta Llama & Community Merge
run_eval "google/gemma-3-27b-it"
run_eval "meta-llama/Llama-3.3-70B-Instruct"
run_eval "mistralai/Ministral-3-14B-Instruct-2512"
run_eval "mistralai/Mixtral-8x7B-Instruct-v0.1"
run_eval "mistralai/Mixtral-8x22B-Instruct-v0.1"

# --- Final Global Stats ---
output=$(uv run exp4_stats.py --indir data --glob "prompt_field_similarity_*.csv")

CLEAN_STATS="${output//\`/’}"
send_telegram "📊 ALL MODELS COMPLETE. Final Stats:\n\`\`\`\n$CLEAN_STATS\n\`\`\`"

echo "Done"
