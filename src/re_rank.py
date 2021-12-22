# -*- encoding:utf-8 -*-
from tqdm import tqdm 
from loguru import logger
import argparse
from collections import defaultdict
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
import os
import torch
import json

def rank(query_embedding, passage_embeddings, k):
    cosine_scores = cos_sim(query_embedding, passage_embeddings).squeeze()
    ranked_scores, ranked_indexes = torch.topk(cosine_scores, k)
    return ranked_scores.tolist(), ranked_indexes.tolist()
        

def main(args):
    model = SentenceTransformer(args.best_model_path)
    query_passage_dict = defaultdict(list)
    test_map = json.load(open(args.test_map_path))
    logger.info('building query passage dict ...')
    out = open(os.path.join(args.result_output_path, args.file_name), 'w+')

    q_id_set, p_id_set = set(), set()

    for line in tqdm(open(args.test_file)):
        q_id, p_id, query, passage = line.strip().split('\t')
        q_id_set.add(q_id); p_id_set.add(p_id)
        query_passage_dict[q_id].append(p_id)

    queries, passages = [], []
    for q_id in q_id_set:
        queries.append(test_map[q_id])
    for p_id in p_id_set:
        passages.append(test_map[p_id])

    logger.info('encoding queries ... ')
    queries_embeddings = model.encode(queries)

    logger.info('encoding passages ... ')
    passages_embeddings = model.encode(passages)
    
    logger.info('building id to embedding dict ... ')
    queryid2embedding, passageid2embedding = {}, {}
    for q_id, q_embedding in zip(list(q_id_set), queries_embeddings):
        queryid2embedding[q_id] = torch.FloatTensor(q_embedding)
    for p_id, p_embedding in zip(list(p_id_set), passages_embeddings):
        passageid2embedding[p_id] = torch.FloatTensor(p_embedding)

    logger.info('re_ranking ...')
    for q_id, p_id_list in tqdm(query_passage_dict.items()):
        cur_q_rank = 0
        query_embedding = queryid2embedding[q_id]
        passage_embeddings = torch.stack([passageid2embedding[p_id] for p_id in p_id_list])
        ranked_scores, ranked_indexes = rank(query_embedding, passage_embeddings, len(p_id_list))
        for score, index in zip(ranked_scores, ranked_indexes):
            cur_q_rank += 1
            cur_p_id = p_id_list[index]
            out.write(f'{q_id} Q0 {cur_p_id} {cur_q_rank} {score} SEN-BERT' + '\n')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_path', type=str)
    parser.add_argument('--result_output_path', type=str)
    parser.add_argument('--test_map_path', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()
    main(args)
    
