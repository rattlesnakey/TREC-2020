#! bin/bash
set -v
set -e
export CUDA_VISIBLE_DEVICES=6
BEST_MODEL_PATH=../models/saved_models
RESULT_OUTPUT_PATH=../results
TEST_MAP_PATH=../data/processed_data/test_map.json
TEST_FILE=../data/raw_data/msmarco-passagetest2020-54-top1000.tsv
FILE_NAME=re_rank_result.txt
python -u ../src/re_rank.py 2>&1 --best_model_path ${BEST_MODEL_PATH} \
    --result_output_path ${RESULT_OUTPUT_PATH} \
    --test_map_path ${TEST_MAP_PATH} \
    --test_file ${TEST_FILE} \
    --file_name ${FILE_NAME} \
    | tee ../logs/re_ranking.log 