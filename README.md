>[比赛网址](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020)
# Model

> 我所用的模型是Sentence-BERT, [论文地址](https://arxiv.org/abs/1908.10084)

* Sentence-BERT采用的是SNLI(Stanford Nature Language Inference)数据，数据集样式如下:

![](https://s2.loli.net/2021/12/22/nH91MKwD2V5FyEN.jpg)

* 其以entailment为positive(相关), contradiction为negative(不相关)来构造一个用来训练衡量句子相似性的三元组数据集: (sentence, relative sentence, unrelative sentence)来对模型进行预训练。

* Sentence-BERT采用的是Siamese Network架构，通过BERT编码器将句子进行编码后，利用BERT pooler 后的结果([CLS] token、Average-strategy、Max-strategy)来代表编码后的句子，然后以原来的sentence 为anchor, 让relative sentence与anchor的cosine similarity越大越好，让unrelative sentence与anchor的cosine similarity越小越好来构造一个triplet loss，以此来进行训练得到。模型架构如下:

<img src="https://s2.loli.net/2021/12/22/GdV3SxwizapFlsZ.jpg" style="zoom:50%;" />



* 在训练中用的是`MultipleNegativesRankingLoss with Hard Negatives`，具体可[参考](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/nli#multiplenegativesrankingloss-with-hard-negatives)，Loss设计图如下

<img src="https://s2.loli.net/2021/12/22/yWXMGQ7wDNOl6oY.jpg" style="zoom:50%;" />

* 输入的数据是triple tuple的形式:`[(a1, b1, c1), ..., (an, bn, cn)]`, 每个tuple里面是一组数据，训练过程中尽可能让`a1` 和 `b1`在 vector space里尽可能接近，让a1 和 c1尽可能不接近, 以这样的方式来fine tuning Sentence-BERT



# Document

```markdown
├── data
│   ├── processed_data
│   └── raw_data
├── logs
│   ├── evaluating.log
│   ├── re_ranking.log
│   └── training.log
├── models
│   ├── pretrained_models
│   └── saved_models
├── results
│   ├── eval
│   ├── Information-Retrieval_evaluation_test_results.csv
│   └── re_rank_result.txt
├── scripts
│   ├── evaluate.sh
│   ├── process.sh
│   ├── re_rank.sh
│   └── train.sh
└── src
    ├── data_process.py
    ├── evaluate.py
    ├── __pycache__
    ├── re_rank.py
    ├── train.py
    └── utils.py
```

*  data目录下的raw_data是原始文件存放的文件夹，processed_data是处理后的文件存放的文件夹

* models里pretrained_models下放的是预训练的bert模型，saved_models是保存的fine_tune完后的模型

* results中

  * 存放训练过程中验证集的NDCG@10值 - eval
  * 测试数据集的NDCG@10值 - Information-Retrieval_evaluation_test_results.csv
  * re_rank结果 - re_rank_result.txt

* scripts中

  * process.sh - 数据处理的脚本
  * train.sh - fine tuning 模型的脚本
  * evaluate.sh - 得出在测试集上的NDCG@10

  * ra_rank.sh - 得到重排序结果

# Getting Started

* conda create -n IR python==3.7 -y
* pip install -r requirement.txt 
* conda install pytorch==1.7.1 cudatoolkit=10.1 -c pytorch -y

主要采用的是Huggingface-Transformer框架

## Data Process

```bash
cd scripts
bash process.sh
```

* 现在目录里已经有数据集了，若要重新构造则重新运行上述内容
* 数据处理主要是去构造构造id 到 query 和到 passage 的映射词典

## Train

```bash
cd scripts
bash train.sh
```

* 现在目录里已经提供了训练好的模型，故不需要重新训练，若要重新训练记得修改train.sh 里面 CUDA_VISIBLE_DEVICES变量来指定用哪个GPU卡
* 主要是用训练集提供的q_id, pos_pid, neg_pid去构造三元组的`MultipleNegativesRankingLoss with Hard Negatives`，然后用这些数据对模型进行微调，代码参考如下

```python
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
```

模型训练代码如下

```python
				train_loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=dev_evaluator,
                epochs=self.num_epochs,
                evaluation_steps=int(len(train_dataloader)*0.1),
                warmup_steps=warmup_steps,
                output_path=self.model_save_path,
                use_amp=False          
                )
```

* Train阶段训练只用了训练集，验证只用了验证集，具体可以参考train.sh里面参数传入的路径

```shell
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
```



## Evaluate

```
cd scripts
bash evaluate.sh
```

- 记得修改evaluate.sh 里面 CUDA_VISIBLE_DEVICES变量来指定用哪个GPU卡
- 验证测试的时候，用的是Python来计算，NDCG(Normalized Discounted Cumulative Gain) 参考了这个[博客](https://www.cnblogs.com/by-dream/p/9403984.html)里的例子来进行书写代码，先用Cosine-Similarity 来对召回的样本进行排序，然后用排序后的结果以及对应的rating值来算NDCG

代码如下

```python
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
```

* evaluate 阶段只用了2020的数据，具体可以参考evaluate.sh下面传入的路径参数

```shell
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
```



## Re-Rank
```
cd scripts
bash re_rank.sh
```

- 记得修改re_rank.sh 里面 CUDA_VISIBLE_DEVICES变量来指定用哪个GPU卡



# Result

```markdown
174463 Q0 470708 1 0.7808177471160889 SEN-BERT
174463 Q0 470712 2 0.7770977020263672 SEN-BERT
174463 Q0 5688367 3 0.6550983190536499 SEN-BERT
174463 Q0 1024766 4 0.6439977884292603 SEN-BERT
174463 Q0 8304456 5 0.6320627927780151 SEN-BERT
174463 Q0 6083371 6 0.6319926977157593 SEN-BERT
174463 Q0 1788361 7 0.629564642906189 SEN-BERT
174463 Q0 8304458 8 0.6190782189369202 SEN-BERT
174463 Q0 6083370 9 0.599174976348877 SEN-BERT
174463 Q0 3235293 10 0.588320255279541 SEN-BERT
174463 Q0 3350452 11 0.5829412937164307 SEN-BERT
174463 Q0 7089629 12 0.5823706388473511 SEN-BERT
174463 Q0 1788355 13 0.568310022354126 SEN-BERT
174463 Q0 7375891 14 0.5674250721931458 SEN-BERT
.....
```

