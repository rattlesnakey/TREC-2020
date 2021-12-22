# -*- encoding:utf-8 -*-
from sentence_transformers.evaluation import SentenceEvaluator
import torch
from torch import Tensor
import logging
from tqdm import tqdm, trange
from sentence_transformers.util import cos_sim, dot_score
import os
import numpy as np
from typing import List, Tuple, Dict, Set, Callable
from loguru import logger

class InformationRetrievalEvaluator(SentenceEvaluator):
    def __init__(self,
                 queries: Dict[str, str],  #qid => query
                 corpus: Dict[str, str],  #cid => doc
                 relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 corpus_chunk_size: int = 50000,
                 ndcg_at_k: List[int] = [10],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim},       #Score function, higher=more similar
                 main_score_function: str = None
                 ):

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        
        self.ndcg_at_k = ndcg_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model, *args, **kwargs)

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.ndcg_at_k:
                    output_data.append(scores[name]['ndcg@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            #! 改成ndcg, self.ndcg_at_k
            return max([scores[name]['ndcg@k'][max(self.ndcg_at_k)] for name in self.score_function_names])
        else:
            #! 改成ndcg
            return scores[self.main_score_function]['ndcg@k'][max(self.ndcg_at_k)]

    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(self.ndcg_at_k)

        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        queries_result_list[name][query_itr].append({'corpus_id': corpus_id, 'score': score})

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])
        return scores


    def compute_metrics(self, queries_result_list: List[object]):
        ndcg = {k: [] for k in self.ndcg_at_k}
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            #! 改成dict
            query_relevant_docs = self.relevant_docs[query_id]
            for k_val in self.ndcg_at_k:
                #! 这边改成rating
                predicted_relevance = [query_relevant_docs[top_hit['corpus_id']] if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                #! 按rating 降序排列
                true_relevances = sorted(predicted_relevance, reverse=True)
                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)
        for k in ndcg:
            ndcg[k] = np.nanmean(ndcg[k])
        return {'ndcg@k': ndcg}


    def output_scores(self, scores):
        for k in scores['ndcg@k']:
            logger.info("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))
            
    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  
        return dcg














