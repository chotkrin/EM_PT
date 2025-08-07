#!/bin/bash

# =============================================================================
# Evaluation Script for Multiple Tasks
# =============================================================================
# This script runs evaluation tasks for various benchmarks including:
# - Math tasks: math500, amc, aime, qwen
# - Code tasks: leetcode, livecodebench
# - Or all tasks combined
# =============================================================================

set -e  # Exit on any error

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# =============================================================================
# Default Configuration
# =============================================================================
MODEL_CKPT=""
MODE=""
TEMP=""
TASK=""
HYPERPARAMETERS=""
NUM_PROCESSES=4
N_TRAJS=4
HELP=false

# =============================================================================
# Helper Functions
# =============================================================================

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

REQUIRED OPTIONS:
    --model_ckpt MODEL          Path to the model checkpoint
    --mode MODE                 Inference mode
    --temp TEMPERATURE          Temperature for sampling
    --task TASK                 Task(s) to run: math, code, all, specific task names, or comma-separated list

OPTIONAL OPTIONS:
    --hyperparameters PARAMS     Hyperparameters (default: empty)
    --num_processes NUM         Number of processes (default: 4)
    --n_trajs NUM              Number of trajectories (default: 4)
    --help                     Show this help message

EXAMPLES:
    $0 --model_ckpt /path/to/model --mode normal --temp 0.7 --task math
    $0 --model_ckpt /path/to/model --mode em_inf --temp 0.7 --task all --num_processes 8
    $0 --model_ckpt /path/to/model --mode self_consistency --temp 0.7 --task amc,math500,leetcode
    $0 --model_ckpt /path/to/model --mode adaptive_temp --temp 0.7 --task "amc, math500, leetcode"

AVAILABLE TASKS:
    math        - Run all math tasks (math500, amc, aime, qwen)
    code        - Run all code tasks (leetcode, livecodebench)
    all         - Run all available tasks
    math500     - Run math500 benchmark only
    amc         - Run AMC benchmark only
    aime        - Run AIME benchmark only
    qwen        - Run Qwen math evaluation only
    leetcode    - Run LeetCode benchmark only
    livecodebench - Run LiveCodeBench only

MULTIPLE TASKS:
    You can specify multiple tasks in several ways:
    - Comma-separated: --task amc,math500,leetcode
    - With spaces: --task "amc, math500, leetcode"
    - Mixed with groups: --task "math, leetcode"
EOF
}

validate_required_args() {
    local missing_args=()
    
    [[ -z "$MODEL_CKPT" ]] && missing_args+=("--model_ckpt")
    [[ -z "$MODE" ]] && missing_args+=("--mode")
    [[ -z "$TEMP" ]] && missing_args+=("--temp")
    [[ -z "$TASK" ]] && missing_args+=("--task")
    
    if [[ ${#missing_args[@]} -gt 0 ]]; then
        echo "Error: Missing required arguments: ${missing_args[*]}" >&2
        echo "Use --help for usage information." >&2
        exit 1
    fi
}

validate_task() {
    local valid_tasks=("math" "code" "all" "math500" "amc" "aime" "qwen" "leetcode" "livecodebench")
    
    # Parse comma-separated tasks
    IFS=',' read -ra task_list <<< "$TASK"
    
    # Validate each task
    for task in "${task_list[@]}"; do
        # Trim whitespace
        task=$(echo "$task" | xargs)
        local task_valid=false
        
        for valid_task in "${valid_tasks[@]}"; do
            if [[ "$task" == "$valid_task" ]]; then
                task_valid=true
                break
            fi
        done
        
        if [[ "$task_valid" == false ]]; then
            echo "Error: Invalid task '$task'. Valid tasks are: ${valid_tasks[*]}" >&2
            exit 1
        fi
    done
}

setup_directories() {
    local repo_dir="$(dirname "$(dirname "$(realpath "$0")")")"
    local model_name="${MODEL_CKPT//\//__}"
    local output_dir="$repo_dir/results/$model_name/$MODE"
    
    mkdir -p "$repo_dir/results"
    mkdir -p "$output_dir"
    
    echo "$repo_dir"
}

get_task_array() {
    local task_input="$1"
    local -n array_ref=$2
    local final_tasks=()
    
    # Parse comma-separated tasks
    IFS=',' read -ra task_list <<< "$task_input"
    
    # Process each task and expand groups
    for task in "${task_list[@]}"; do
        # Trim whitespace
        task=$(echo "$task" | xargs)
        
        case "$task" in
            "math")
                final_tasks+=(math500 amc aime qwen)
                ;;
            "code")
                final_tasks+=(leetcode livecodebench)
                ;;
            "all")
                final_tasks+=(math500 amc aime qwen leetcode livecodebench)
                ;;
            *)
                final_tasks+=("$task")
                ;;
        esac
    done
    
    # Remove duplicates while preserving order
    local -A seen
    array_ref=()
    for task in "${final_tasks[@]}"; do
        if [[ -z "${seen[$task]}" ]]; then
            seen[$task]=1
            array_ref+=("$task")
        fi
    done
}

print_configuration() {
    local model_name="${MODEL_CKPT//\//__}"
    local output_dir="$1"
    local -n task_array_ref=$2
    
    echo "****************************************************************************************************************************************************************"
    echo "CONFIGURATION SUMMARY:"
    echo "  Model Checkpoint: $MODEL_CKPT"
    echo "  Model Name: $model_name"
    echo "  Mode: $MODE"
    echo "  Temperature: $TEMP"
    echo "  Task Input: $TASK"
    echo "  Tasks to Execute: ${task_array_ref[*]}"
    echo "  Number of Trajectories: $N_TRAJS"
    echo "  Number of Processes: $NUM_PROCESSES"
    echo "  Hyperparameters: $HYPERPARAMETERS"
    echo "  Output Directory: $output_dir"
    echo "****************************************************************************************************************************************************************"
}

# =============================================================================
# Task Execution Functions
# =============================================================================

run_leetcode() {
    local output_dir="$1"
    
    echo "Running LeetCode evaluation..."
    # source virtual_env/prime/bin/activate
    source "/work/nvme/bcaq/zzhang32/env/prime/bin/activate"
    
    mkdir -p "$output_dir/leetcode_chat"
    python3 Coding/leetcode/evaluate_leetcode.py \
        --model "$MODEL_CKPT" \
        --input_data data/leetcode/leetcode-test.json \
        --save_dir "$output_dir/leetcode_chat" \
        --temperature "$TEMP" \
        --hyperparameters "$HYPERPARAMETERS" \
        --inference_mode "$MODE"
}

run_amc() {
    local output_dir="$1"
    
    echo "Running AMC evaluation (numina)..."
    # source virtual_env/prime/bin/activate
    source /work/nvme/bcaq/zzhang32/env/prime/bin/activate
    
    mkdir -p "$output_dir/amc_chat"
    python3 -u Math/amc/evaluate_amc.py \
        --model "$MODEL_CKPT" \
        --data_dir data/AI-MO/aimo-validation-amc \
        --save_dir "$output_dir/amc_chat" \
        --inference_mode "$MODE" \
        --temp "$TEMP" \
        --hyperparameters "$HYPERPARAMETERS" \
        --n_trajs "$N_TRAJS" \
        --num_processes "$NUM_PROCESSES"
}

run_aime() {
    local output_dir="$1"
    
    echo "Running AIME evaluation (numina)..."
    # source virtual_env/prime/bin/activate
    source /work/nvme/bcaq/zzhang32/env/prime/bin/activate
    
    mkdir -p "$output_dir/aime_chat"
    python3 -u Math/aime/evaluate_aime.py \
        --model "$MODEL_CKPT" \
        --data_dir data/AI-MO/aimo-validation-aime \
        --save_dir "$output_dir/aime_chat" \
        --inference_mode "$MODE" \
        --temp "$TEMP" \
        --hyperparameters "$HYPERPARAMETERS" \
        --n_trajs "$N_TRAJS" \
        --num_processes "$NUM_PROCESSES"
}

run_math500() {
    local output_dir="$1"
    
    echo "Running Math500 evaluation..."
    # source virtual_env/prime/bin/activate
    source /work/nvme/bcaq/zzhang32/env/prime/bin/activate
    
    mkdir -p "$output_dir/math_chat"
    python3 -u Math/math/evaluate_math.py \
        --model "$MODEL_CKPT" \
        --data_dir data/math500 \
        --save_dir "$output_dir/math_chat" \
        --inference_mode "$MODE" \
        --temp "$TEMP" \
        --hyperparameters "$HYPERPARAMETERS" \
        --n_trajs "$N_TRAJS" \
        --num_processes "$NUM_PROCESSES"
}

run_qwen() {
    local output_dir="$1"
    local current_dir="$PWD"
    
    echo "Running Qwen math evaluation..."
    # source virtual_env/qwen_math/bin/activate
    source /work/nvme/bcaq/zzhang32/env/qwen_math/bin/activate
    
    mkdir -p "$output_dir/qwen_math"

    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u Math/Qwen25-Math/math_eval.py \
        --model_name_or_path $MODEL_CKPT \
        --data_name "minerva_math,olympiadbench" \
        --output_dir $output_dir/qwen_math \
        --split test \
        --prompt_type "qwen25-math-cot" \
        --num_test_sample -1 \
        --seed 42 \
        --temperature ${TEMP} \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --overwrite \
        --apply_chat_template \
        --inference_mode ${MODE} \
        --hyperparameters "${HYPERPARAMETERS}" \
        --num_processes ${NUM_PROCESSES}
    
    # bash sh/eval.sh "$prompt_type" "$MODEL_CKPT" "$output_dir/qwen_math" "$MODE" "$TEMP" "$HYPERPARAMETERS" "$NUM_PROCESSES"
    
    cd "$current_dir"
}

run_livecodebench() {
    local output_dir="$1"
    local current_dir="$PWD"
    
    echo "Running LiveCodeBench evaluation..."
    # source virtual_env/lcb/bin/activate
    source /work/nvme/bcaq/zzhang32/env/lcb/bin/activate
    
    cd ./Coding/livecodebench/LiveCodeBench-main
    mkdir -p "$output_dir/livecodebench"
    
    python -m lcb_runner.runner.main \
        --model "$MODEL_CKPT" \
        --scenario codegeneration \
        --evaluate \
        --release_version release_v2 \
        --output_path "$output_dir/livecodebench" \
        --n 1 \
        --tensor_parallel_size 4
    
    # Compute scores for v2
    nohup python -m lcb_runner.evaluation.compute_scores \
        --eval_all_file "$output_dir/livecodebench/result_eval_all.json" \
        --start_date 2023-05-01 \
        --end_date 2024-05-31 \
        > "$output_dir/livecodebench/lcb_v2.txt" 2>&1 &
    
    cd "$current_dir"
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model_ckpt)
                [[ -z "$2" ]] && { echo "Error: --model_ckpt requires a value" >&2; exit 1; }
                MODEL_CKPT="$2"
                shift 2
                ;;
            --mode)
                [[ -z "$2" ]] && { echo "Error: --mode requires a value" >&2; exit 1; }
                MODE="$2"
                shift 2
                ;;
            --temp)
                [[ -z "$2" ]] && { echo "Error: --temp requires a value" >&2; exit 1; }
                TEMP="$2"
                shift 2
                ;;
            --task)
                [[ -z "$2" ]] && { echo "Error: --task requires a value" >&2; exit 1; }
                TASK="$2"
                shift 2
                ;;
            --hyperparameters)
                [[ -z "$2" ]] && { echo "Error: --hyperparameters requires a value" >&2; exit 1; }
                HYPERPARAMETERS="$2"
                shift 2
                ;;
            --num_processes)
                [[ -z "$2" ]] && { echo "Error: --num_processes requires a value" >&2; exit 1; }
                NUM_PROCESSES="$2"
                shift 2
                ;;
            --n_trajs)
                [[ -z "$2" ]] && { echo "Error: --n_trajs requires a value" >&2; exit 1; }
                N_TRAJS="$2"
                shift 2
                ;;
            --help|-h)
                HELP=true
                shift
                ;;
            *)
                echo "Error: Unknown argument '$1'" >&2
                echo "Use --help for usage information." >&2
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Show help if requested
    if [[ "$HELP" == true ]]; then
        show_help
        exit 0
    fi
    
    # Validate arguments
    validate_required_args
    validate_task
    
    # Setup directories and get configuration
    local repo_dir
    repo_dir=$(setup_directories)
    local model_name="${MODEL_CKPT//\//__}"
    local output_dir="$repo_dir/results/$model_name/$MODE"
    
    # Determine which tasks to run
    local task_array=()
    get_task_array "$TASK" task_array
    
    # Print configuration summary
    print_configuration "$output_dir" task_array
    
    # Change to evaluation directory
    cd "$repo_dir/eval"
    
    # Execute tasks
    for task in "${task_array[@]}"; do
        case "$task" in
            "leetcode")
                run_leetcode "$output_dir"
                ;;
            "amc")
                run_amc "$output_dir"
                ;;
            "aime")
                run_aime "$output_dir"
                ;;
            "math500")
                run_math500 "$output_dir"
                ;;
            "qwen")
                run_qwen "$output_dir"
                ;;
            "livecodebench")
                run_livecodebench "$output_dir"
                ;;
            *)
                echo "Warning: Unknown task '$task', skipping..." >&2
                ;;
        esac
    done
    
    echo "All requested tasks completed successfully!"
}

# Run main function with all arguments
main "$@"