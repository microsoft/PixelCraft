#!/bin/bash
set -e
source .venv/bin/activate

export OPENAI_API_KEY=sk-xxx
export BASE_URL="api_url"
export DEPLOYMENT_NAME="gpt-4.1-mini-2025-04-14"

exp_name=eval
run_id=1205
model_name=${DEPLOYMENT_NAME:-gpt-4.1-mini-2025-04-14}

mkdir -p logs/${exp_name}

log_file=logs/${exp_name}/${run_id}.log
echo "Eval CharXiv ${run_id} using ${model_name}" | tee ${log_file}

python -u src/core/processor.py \
    --data_path data/chartxiv_val.json \
    --data_dir_root ./ \
    --output_dir_root output/${exp_name} \
    --tool_selection_path data/charxiv_selection.json \
    --output_dir "${run_id}" \
    --model_name "${model_name}" \
    --max_concurrent 8 2>&1 | tee -a ${log_file} 
