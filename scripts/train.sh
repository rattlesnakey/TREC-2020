#! bin/bash
set -v
set -e
export CUDA_VISIBLE_DEVICES=6

PREFIX_PATH=../data/processed_data
MODEL_NAME=bert-base-uncased
PRETRAINED_MODEL_PATH=../models/pretrained_models/bert-base-uncased
TRAIN_BATCH_SIZE=32
MAX_SEQ_LENGTH=75
NUM_EPOCHS=10
TRAIN_TRIPLE_FILE_PATH=../data/raw_data/qidpidtriples.train.sampled.tsv
DEV_DATA_PATH=../data/raw_data/2019qrels-pass.txt
MODEL_SAVE_PATH=../models/saved_models
    
python -u ../src/train.py 2>&1 --prefix_path ${PREFIX_PATH}\
    --model_name ${MODEL_NAME} \
    --pretrained_model_path ${PRETRAINED_MODEL_PATH} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --num_epochs ${NUM_EPOCHS} \
    --train_triple_file_path ${TRAIN_TRIPLE_FILE_PATH} \
    --dev_data_path ${DEV_DATA_PATH} \
    --model_save_path ${MODEL_SAVE_PATH} \
    | tee ../logs/training.log 

mv ${MODEL_SAVE_PATH}/eval ../results