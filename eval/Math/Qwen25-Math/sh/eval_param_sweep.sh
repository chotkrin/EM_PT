set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=$3
MODE=$4
TEMP=$5
MIN_ENT_PARAMS_STRING="$6"
NUM_PROCESSES=$7
SPLIT="test"
NUM_TEST_SAMPLE=-1

echo $MIN_ENT_PARAMS_STRING

# English open datasets
DATA_NAME="minerva_math,olympiadbench"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
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
    --min_ent_params "${MIN_ENT_PARAMS_STRING}" \
    --num_processes ${NUM_PROCESSES} \
    --param_sweep