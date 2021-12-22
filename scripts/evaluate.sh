#! bin/bash
set -v
set -e
export CUDA_VISIBLE_DEVICES=6
BEST_MODEL_PATH=../models/saved_models
RESULT_OUTPUT_PATH=../results
TEST_MAP_PATH=../data/processed_data/test_map.json
TEST_DATA_PATH=../data/raw_data/2020qrels-pass.txt
python -u ../src/evaluate.py 2>&1 --best_model_path ${BEST_MODEL_PATH} \
    --result_output_path ${RESULT_OUTPUT_PATH} \
    --test_map_path ${TEST_MAP_PATH} \
    --test_data_path ${TEST_DATA_PATH} \
    | tee ../logs/evaluating.log 