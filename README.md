# ark-nlp
ark-nlp主要是收集和复现学术与工作中常用的NLP模型



## 环境 

* python 3
* torch >= 1.0.0 
* tqdm >= 4.56.0 
* jieba >= 0.42.1 
* transformers >= 3.0.0 
* zhon >= 1.1.5 



## pip安装

```
pip install --upgrade ark-nlp
```



## 项目结构

<table>
<tr>
    <td><b> ark_nlp</b></td>
    <td> 开源的自然语言处理库 </td>
</tr>
<tr>
    <td><b> ark_nlp.dataset </b></td>
    <td> 封装数据加载、处理和转化等功能 </td>
</tr>
<tr>
    <td><b> ark_nlp.nn</b></td>
    <td> 封装一些完整的神经网络模型 </td>
</tr>
<tr>
    <td><b> ark_nlp.processor</b></td>
    <td>  封装分词器、词典和构图器等 </td>
</tr>
<tr>
    <td><b> ark_nlp.factory </b></td>
    <td> 封装损失函数、优化器、训练和预测等功能 </td>
</tr>
<tr>
    <td><b>ark_nlp.model</b></td>
    <td> 按实际NLP任务封装常用的模型，方便调用</td>
</tr>
</table>



## 实现的模型

### 预训练模型

| 模型     | 简介 |
| :------- | ---- |
| BERT     |      |
| ERNIE1.0 |      |
| NEZHA    |      |
| Roformer |      |

### 文本分类 (Text Classification)
| 模型             | 简介                                          |
| :--------------- | --------------------------------------------- |
| RNN/CNN/GRU/LSTM | 经典的RNN, CNN, GRU, LSTM等经典文本分类结构 |
| BERT/ERNIE       | 常用的预训练模型分类                        |


### 文本匹配 (Text Matching)

| 模型       | 简介                       |
| :--------- | -------------------------- |
| BERT/ERNIE | 常用的预训练模型匹配分类 |

### 命名实体识别 (Named Entity Recognition)

| 模型       | 简介                       |
| :--------- | -------------------------- |
| CRF BERT |  |
| Biaffine BERT |  |
| Span BERT |  |
| Global Pointer BERT |  |

### 关系抽取 (Relation Extraction)

| 模型       | 简介                       |
| :--------- | -------------------------- |
| Casrel | A Novel Cascade Binary Tagging Framework for Relational Triple Extraction|
| PRGC | PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction |

## 使用例子



* 文本分类

  ```python
  import torch
  import pandas as pd
  
  from ark_nlp.model.tc.bert import Bert
  from ark_nlp.model.tc.bert import BertConfig
  from ark_nlp.model.tc.bert import Dataset
  from ark_nlp.model.tc.bert import Task
  from ark_nlp.model.tc.bert import get_default_model_optimizer
  from ark_nlp.model.tc.bert import Tokenizer
  
  # 加载数据集
  tc_train_dataset = Dataset(train_data_df)
  tc_dev_dataset = Dataset(dev_data_df)
  
  # 加载分词器
  tokenizer = Tokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=30)
  
  # 加载预训练模型
  config = BertConfig.from_pretrained('nghuyong/ernie-1.0', 
                                       num_labels=len(tc_train_dataset.cat2id))
  dl_module = Bert.from_pretrained('nghuyong/ernie-1.0', 
                                   config=config)
  
  # 任务构建
  num_epoches = 10
  batch_size = 32
  optimizer = get_default_model_optimizer(dl_module)
  model = Task(dl_module, optimizer, 'ce', cuda_device=0)
  
  # 训练
  model.fit(tc_train_dataset, 
            tc_dev_dataset,
            lr=2e-5,
            epochs=5, 
            batch_size=batch_size
           )
  
  # 推断
  from ark_nlp.model.tc.bert import Predictor
  
  tc_predictor_instance = Predictor(model.module, tokenizer, tc_train_dataset.cat2id)
  
  tc_predictor_instance.predict_one_sample(待预测文本)
  ```

## PS

  ```
  本项目用于收集和复现学术与工作中常用的NLP模型，整合成方便调用的形式，所以参考借鉴了网上很多开源实现，如有不当的地方，还请联系批评指教。
  在此，感谢大佬们的开源实现。
  ```
