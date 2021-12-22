# -*- encoding:utf-8 -*-
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from datetime import datetime
from utils import InformationRetrievalEvaluator
import sys
import os
import random
from loguru import logger
import json
from tqdm import tqdm
import argparse

class Train(object):
    def __init__(self,args):
        self.prefix_path = args.prefix_path
        self.model_name = args.model_name
        self.pretrained_model_path = args.pretrained_model_path
        self.train_batch_size = args.train_batch_size
        self.max_seq_length = args.max_seq_length
        self.num_epochs = args.num_epochs
        self.train_triple_file_path = args.train_triple_file_path
        self.dev_data_path = args.dev_data_path
        self.model_save_path = args.model_save_path

    
    def __load_map(self, ):
        train_id2passage, train_id2query, valid_map = [json.load(open(os.path.join(self.prefix_path, file_name),'r')) for file_name in ['train_id2passage.json', 'train_id2query.json', 'valid_map.json']]
        return train_id2passage, train_id2query, valid_map
    
    def __build_model(self, ):
        word_embedding_model = models.Transformer(self.pretrained_model_path, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        return model

    def __build_train_dataloader(self, train_id2query, train_id2passage, train_triple_file_path):
        train_samples = []
        train_triple_file = open(train_triple_file_path)
        for line in tqdm(train_triple_file):
            qid, pos_pid, neg_pid = line.strip().split('\t')
            query = train_id2query[qid]
            pos_passage = train_id2passage[pos_pid]
            neg_passage = train_id2passage[neg_pid]
            train_samples.append(InputExample(texts=[query, pos_passage, neg_passage]))
        
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=self.train_batch_size)
        return train_dataloader

    def __build_dev_evaluator(self, dev_map, dev_data_path):
        dev_data = open(dev_data_path, 'r')
        qid2query, pid2passage, relevant_docs = {}, {}, {}
        for line in tqdm(dev_data):
            try:
                qid, split_signal, pid, score = line.strip().split()
                query = dev_map[qid]
                passage = dev_map[pid]
                qid2query[qid] = query
                pid2passage[pid] = passage
                if qid not in relevant_docs:
                    relevant_docs[qid] = dict()
                relevant_docs[qid][pid] = float(score)
            except KeyError:
                continue
        dev_evaluator = InformationRetrievalEvaluator(queries=qid2query, corpus=pid2passage, relevant_docs=relevant_docs, ndcg_at_k=[10], name='dev', show_progress_bar=True, )
        return dev_evaluator
    
    def __call__(self, ):
        logger.info('loading map ...')
        train_id2passage, train_id2query, valid_map = self.__load_map()
        logger.info('building model ...')
        model = self.__build_model()
        logger.info('building train dataloader ...')
        train_dataloader = self.__build_train_dataloader(train_id2query, train_id2passage, self.train_triple_file_path)
        logger.info('building dev dataloader ...')
        dev_evaluator = self.__build_dev_evaluator(valid_map, self.dev_data_path)
        warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1) 
        logger.info("Warmup-steps: {}".format(warmup_steps))
        train_loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=dev_evaluator,
                epochs=self.num_epochs,
                evaluation_steps=int(len(train_dataloader)*0.1),
                warmup_steps=warmup_steps,
                output_path=self.model_save_path,
                use_amp=False          
                )
        logger.info("finish training!")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--train_triple_file_path', type=str)
    parser.add_argument('--dev_data_path', type=str)
    parser.add_argument('--model_save_path', type=str)
    args = parser.parse_args()
    t = Train(args)
    sys.exit(t())