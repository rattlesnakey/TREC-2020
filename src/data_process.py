# -*- encoding:utf-8 -*-
from loguru import logger
from tqdm import tqdm
import json
import os
import sys
import argparse

class DataProcess(object):
    def __init__(self, args):  
        self.train_id_passage_path = args.train_id_passage_path
        self.train_id_query_path = args.train_id_query_path
        self.valid_qid_pid_query_passage = args.valid_qid_pid_query_passage
        self.test_qid_pid_query_passage = args.test_qid_pid_query_passage

    def __build_mapping(self, dataset_type, content_type):
        cur_map = {}
        if dataset_type == 'train':
            logger.info(f"building {dataset_type} {content_type}'s mapping ...")
            if content_type == 'passage':
                cur_file = self.train_id_passage_path
            else: 
                cur_file = self.train_id_query_path
            for line in tqdm(open(cur_file)):
                id, content = line.strip().split('\t')
                cur_map[id] = content
        else:
            logger.info(f'building mapping {dataset_type}...')
            if dataset_type == 'valid':
                cur_file = self.valid_qid_pid_query_passage
            else:
                cur_file = self.test_qid_pid_query_passage

            for line in tqdm(open(cur_file)):
                qid, pid, query, passage = line.strip().split('\t')
                cur_map[qid], cur_map[pid] = query, passage
        return cur_map

    def __call__(self, ):
        train_id2passage = self.__build_mapping('train', 'passage')
        train_id2query = self.__build_mapping('train', 'query')
        valid_map = self.__build_mapping('valid', None)
        test_map = self.__build_mapping('test', None)

        for mapping, file_name in zip([train_id2passage, train_id2query, valid_map, test_map],['train_id2passage.json', 'train_id2query.json', 'valid_map.json', 'test_map.json']):
            json.dump(mapping, open(os.path.join('../data/processed_data', file_name),'w+'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_id_passage_path', type=str)
    parser.add_argument('--train_id_query_path', type=str)
    parser.add_argument('--valid_qid_pid_query_passage', type=str)
    parser.add_argument('--test_qid_pid_query_passage', type=str)
    args = parser.parse_args()
    p = DataProcess(args)
    sys.exit(p())
    logger.info('finish!')