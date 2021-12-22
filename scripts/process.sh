#! bin/bash
set -v
set -e

PREFIX=../data/raw_data
python ../src/data_process.py \
    --train_id_passage_path ${PREFIX}/collection.train.sampled.tsv \
    --train_id_query_path ${PREFIX}/queries.train.sampled.tsv \
    --valid_qid_pid_query_passage ${PREFIX}/msmarco-passagetest2019-43-top1000.tsv \
    --test_qid_pid_query_passage ${PREFIX}/msmarco-passagetest2020-54-top1000.tsv 
    