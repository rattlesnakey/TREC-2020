# -*- encoding:utf-8 -*-
import json
from utils import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger
import argparse

def evaluate(args):
    test_map = json.load(open(args.test_map_path))
    test_data = open(args.test_data_path)
    qid2query, pid2passage, relevant_docs = {}, {}, {}
    logger.info("building test dataset ...")
    for line in tqdm(test_data):
        try:
            qid, split_signal, pid, score = line.strip().split()
            query = test_map[qid]
            passage = test_map[pid]
            qid2query[qid] = query
            pid2passage[pid] = passage
            if qid not in relevant_docs:
                relevant_docs[qid] = dict()
            relevant_docs[qid][pid] = float(score)
        except KeyError:
            continue
    logger.info("loading best_model ...")
    model = SentenceTransformer(args.best_model_path)
    test_evaluator = InformationRetrievalEvaluator(queries=qid2query, corpus=pid2passage, relevant_docs=relevant_docs, ndcg_at_k=[10], name='test', show_progress_bar=True, )
    logger.info("start testing ...")
    test_evaluator(model, output_path=args.result_output_path)



# model_save_path = '../models/saved_models/fine_tuned'
# result_output_path = '../results'
# test_map = json.load(open('../data/processed_data/test_map.json'))
# logger.info("building test dataset ...")
# test_data_path = '../data/raw_data/2020qrels-pass.txt'
# test_data = open(test_data_path, 'r')
# qid2query, pid2passage, relevant_docs = {}, {}, {}

# for line in tqdm(test_data):
#     #! 存在有的id找不到，数据问题 
#     try:
#         qid, split_signal, pid, score = line.strip().split()
#         query = test_map[qid]
#         passage = test_map[pid]
#         qid2query[qid] = query
#         pid2passage[pid] = passage
#         if qid not in relevant_docs:
#             relevant_docs[qid] = dict()
#         relevant_docs[qid][pid] = float(score)
#     except KeyError:
#         continue
# model = SentenceTransformer(model_save_path)
# ndcg_at_k=[10]
# test_evaluator = InformationRetrievalEvaluator(queries=qid2query, corpus=pid2passage, relevant_docs=relevant_docs, ndcg_at_k=[10], name='test', show_progress_bar=True, )
# test_evaluator(model, output_path=result_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_path', type=str)
    parser.add_argument('--result_output_path', type=str)
    parser.add_argument('--test_map_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    args = parser.parse_args()
    evaluate(args)