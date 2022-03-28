# ark-nlp
ark-nlp主要是收集和复现学术与工作中常用的NLP模型



## 环境 

* python 3
* torch >= 1.0.0, <1.10.0
* tqdm >= 4.56.0 
* jieba >= 0.42.1 
* transformers >= 3.0.0 
* zhon >= 1.1.5 
* scipy >= 1.2.0
* scikit-learn >= 0.17.0



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

|   模型   |                           参考文献                           |
| :------: | :----------------------------------------------------------: |
|   BERT   | [BERT:Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) |
| ERNIE1.0 | [ERNIE:Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223.pdf) |
|  NEZHA   | [NEZHA:Neural Contextualized Representation For Chinese Language Understanding](https://arxiv.org/pdf/1909.00204.pdf) |
| Roformer | [Roformer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf) |

### 文本分类 (Text Classification)
|       模型       |                    简介                     |
| :--------------: | :-----------------------------------------: |
| RNN/CNN/GRU/LSTM | 经典的RNN, CNN, GRU, LSTM等经典文本分类结构 |
|    BERT/ERNIE    |            常用的预训练模型分类             |


### 文本匹配 (Text Matching)

| 模型       | 简介                       |
| :--------: | :------------------------: |
| [BERT/ERNIE](https://github.com/xiangking/ark-nlp/blob/main/test/text_match.ipynb) | 常用的预训练模型匹配分类 |
| [UnsupervisedSimcse](https://github.com/xiangking/ark-nlp/blob/main/test/simcse.ipynb) | 无监督Simcse匹配算法 |
| [CoSENT](https://github.com/xiangking/PyTorch_CoSENT) | [CoSENT：比Sentence-BERT更有效的句向量方案](https://spaces.ac.cn/archives/8847) |


### 命名实体识别 (Named Entity Recognition)

| 模型       | 简介                       |
| :--------: | :------------------------: |
| [CRF BERT](https://github.com/xiangking/ark-nlp/blob/main/test/crf_bert.ipynb) |  |
| [Biaffine BERT](https://github.com/xiangking/ark-nlp/blob/main/test/biaffine_bert.ipynb) |  |
| [Span BERT](https://github.com/xiangking/ark-nlp/blob/main/test/span_bert.ipynb) |  |
| [Global Pointer BERT](https://github.com/xiangking/ark-nlp/blob/main/test/gobalpoint_bert.ipynb) | [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://www.kexue.fm/archives/8373) |
| [Efficient Global Pointer BERT](https://github.com/xiangking/ark-nlp/blob/main/ark_nlp/model/ner/global_pointer_bert/global_pointer_bert.py) | [Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877) |

### 关系抽取 (Relation Extraction)

|                             模型                             |                           参考文献                           |                   论文源码                    |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-------------------------------------------: |
| [Casrel](https://github.com/xiangking/ark-nlp/blob/main/test/casrel_bert.ipynb) | [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction](https://arxiv.org/pdf/1909.03227.pdf) | [github](https://github.com/weizhepei/CasRel) |
| [PRGC](https://github.com/xiangking/ark-nlp/blob/main/test/prgc_bert.ipynb) | [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction](https://arxiv.org/pdf/2106.09895.pdf) | [github](https://github.com/hy-struggle/PRGC) |

## 实际应用

* [CHIP2021-Task3-临床术语标准化任务-第三名](https://github.com/DataArk/CHIP2021-Task3)
* [CHIP2021-Task1-医学对话临床发现阴阳性判别任务-第一名](https://github.com/DataArk/CHIP2021-Task1-Top1)
* [中文医疗信息处理挑战榜CBLUE](https://github.com/DataArk/CBLUE-Baseline)

## 使用例子

完整代码可参考`test`文件夹

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
  # train_data_df的columns必选包含"text"和"label"
  # text列为文本，label列为分类标签
  tc_train_dataset = Dataset(train_data_df)
  tc_dev_dataset = Dataset(dev_data_df)
  
  # 加载分词器
  tokenizer = Tokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=30)
  
  # 文本切分、ID化
  tc_train_dataset.convert_to_ids(tokenizer)
  tc_dev_dataset.convert_to_ids(tokenizer)
  
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



* 文本匹配

  ```python
  import torch
  import pandas as pd
  
  from ark_nlp.model.tm.bert import Bert
  from ark_nlp.model.tm.bert import BertConfig
  from ark_nlp.model.tm.bert import Dataset
  from ark_nlp.model.tm.bert import Task
  from ark_nlp.model.tm.bert import get_default_model_optimizer
  from ark_nlp.model.tm.bert import Tokenizer
  
  # 加载数据集
  # train_data_df的columns必选包含"text_a"、"text_b"和"label"
  # text_a和text_b列为文本，label列为匹配标签
  tm_train_dataset = Dataset(train_data_df)
  tm_dev_dataset = Dataset(dev_data_df)
  
  # 加载分词器
  tokenizer = Tokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=30)
  
  # 文本切分、ID化
  tm_train_dataset.convert_to_ids(tokenizer)
  tm_dev_dataset.convert_to_ids(tokenizer)
  
  # 加载预训练模型
  config = BertConfig.from_pretrained('nghuyong/ernie-1.0', 
                                     num_labels=len(tm_train_dataset.cat2id))
  dl_module = Bert.from_pretrained('nghuyong/ernie-1.0', 
                                   config=config)
  
  # 任务构建
  num_epoches = 10
  batch_size = 32
  optimizer = get_default_model_optimizer(dl_module)
  model = Task(dl_module, optimizer, 'ce', cuda_device=0)
  
  # 训练
  model.fit(tm_train_dataset, 
            tm_dev_dataset,
            lr=2e-5,
            epochs=5, 
            batch_size=batch_size
           )
  
  # 推断
  from ark_nlp.model.tm.bert import Predictor
  
  tm_predictor_instance = Predictor(model.module, tokenizer, tm_train_dataset.cat2id)
  
  tm_predictor_instance.predict_one_sample([待预测文本A, 待预测文本B])
  ```



* 命名实体

  ```python
  import torch
  import pandas as pd
  
  from ark_nlp.model.ner.crf_bert import CRFBert
  from ark_nlp.model.ner.crf_bert import CRFBertConfig
  from ark_nlp.model.ner.crf_bert import Dataset
  from ark_nlp.model.ner.crf_bert import Task
  from ark_nlp.model.ner.crf_bert import get_default_model_optimizer
  from ark_nlp.model.ner.crf_bert import Tokenizer
  
  # 加载数据集
  # train_data_df的columns必选包含"text"和"label"
  # text列为文本
  # label列为列表形式，列表中每个元素是如下组织的字典
  # {'start_idx': 实体首字符在文本的位置, 'end_idx': 实体尾字符在文本的位置, 'type': 实体类型标签, 'entity': 实体}
  ner_train_dataset = Dataset(train_data_df)
  ner_dev_dataset = Dataset(dev_data_df)
  
  # 加载分词器
  tokenizer = Tokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=30)
  
  # 文本切分、ID化
  ner_train_dataset.convert_to_ids(tokenizer)
  ner_dev_dataset.convert_to_ids(tokenizer)
  
  # 加载预训练模型
  config = CRFBertConfig.from_pretrained('nghuyong/ernie-1.0', 
                                    num_labels=len(ner_train_dataset.cat2id))
  dl_module = CRFBert.from_pretrained('nghuyong/ernie-1.0', 
                                      config=config)
  
  # 任务构建
  num_epoches = 10
  batch_size = 32
  optimizer = get_default_model_optimizer(dl_module)
  model = Task(dl_module, optimizer, 'ce', cuda_device=0)
  
  # 训练
  model.fit(ner_train_dataset, 
            ner_dev_dataset,
            lr=2e-5,
            epochs=5, 
            batch_size=batch_size
           )
  
  # 推断
  from ark_nlp.model.ner.crf_bert import Predictor
  
  ner_predictor_instance = Predictor(model.module, tokenizer, ner_train_dataset.cat2id)
  
  ner_predictor_instance.predict_one_sample(待抽取文本)
  ```



* Casrel关系抽取

  ```python
  import torch
  import pandas as pd
  
  from ark_nlp.model.re.casrel_bert import CasRelBert
  from ark_nlp.model.re.casrel_bert import CasRelBertConfig
  from ark_nlp.model.re.casrel_bert import Dataset
  from ark_nlp.model.re.casrel_bert import Task
  from ark_nlp.model.re.casrel_bert import get_default_model_optimizer
  from ark_nlp.model.re.casrel_bert import Tokenizer
  from ark_nlp.factory.loss_function import CasrelLoss
  
  # 加载数据集
  # train_data_df的columns必选包含"text"和"label"
  # text列为文本
  # label列为列表形式，列表中每个元素是如下组织的字典
  # [头实体, 头实体首字符在文本的位置, 头实体尾字符在文本的位置, 关系类型, 尾实体, 尾实体首字符在文本的位置, 尾实体尾字符在文本的位置]
  re_train_dataset = Dataset(train_data_df)
  re_dev_dataset = Dataset(dev_data_df,
                           categories = re_train_dataset.categories,
                           is_train=False)
  
  # 加载分词器
  tokenizer = Tokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=100)
  
  # 文本切分、ID化
  # 注意：casrel的代码这部分其实并没有进行切分、ID化，仅是将分词器赋予dataset对象
  re_train_dataset.convert_to_ids(tokenizer)
  re_dev_dataset.convert_to_ids(tokenizer)
  
  # 加载预训练模型
  config = CasRelBertConfig.from_pretrained('nghuyong/ernie-1.0',
                                            num_labels=len(re_train_dataset.cat2id))
  dl_module = CasRelBert.from_pretrained('nghuyong/ernie-1.0', 
                                         config=config)
  
  # 任务构建
  num_epoches = 40
  batch_size = 16
  optimizer = get_default_model_optimizer(dl_module)
  model = Task(dl_module, optimizer, CasrelLoss(), cuda_device=0)
  
  # 训练
  model.fit(re_train_dataset, 
            re_dev_dataset,
            lr=2e-5,
            epochs=5, 
            batch_size=batch_size
           )
  
  # 推断
  from ark_nlp.model.re.casrel_bert import Predictor
  
  casrel_re_predictor_instance = Predictor(model.module, tokenizer, re_train_dataset.cat2id)
  
  casrel_re_predictor_instance.predict_one_sample(待抽取文本)
  ```



* PRGC关系抽取

  ```python
  import torch
  import pandas as pd
  
  from ark_nlp.model.re.prgc_bert import PRGCBert
  from ark_nlp.model.re.prgc_bert import PRGCBertConfig
  from ark_nlp.model.re.prgc_bert import Dataset
  from ark_nlp.model.re.prgc_bert import Task
  from ark_nlp.model.re.prgc_bert import get_default_model_optimizer
  from ark_nlp.model.re.prgc_bert import Tokenizer
  
  # 加载数据集
  # train_data_df的columns必选包含"text"和"label"
  # text列为文本
  # label列为列表形式，列表中每个元素是如下组织的字典
  # [头实体, 头实体首字符在文本的位置, 头实体尾字符在文本的位置, 关系类型, 尾实体, 尾实体首字符在文本的位置, 尾实体尾字符在文本的位置]
  re_train_dataset = Dataset(train_df, is_retain_dataset=True)
  re_dev_dataset = Dataset(dev_df,
                           categories = re_train_dataset.categories,
                           is_train=False)
  
  # 加载分词器
  tokenizer = Tokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=100)
  
  # 文本切分、ID化
  re_train_dataset.convert_to_ids(tokenizer)
  re_dev_dataset.convert_to_ids(tokenizer)
  
  # 加载预训练模型
  config = PRGCBertConfig.from_pretrained('nghuyong/ernie-1.0',
                                            num_labels=len(re_train_dataset.cat2id))
  dl_module = PRGCBert.from_pretrained('nghuyong/ernie-1.0', 
                                         config=config)
  
  # 任务构建
  num_epoches = 40
  batch_size = 16
  optimizer = get_default_model_optimizer(dl_module)
  model = Task(dl_module, optimizer, None, cuda_device=0)
  
  # 训练
  model.fit(re_train_dataset, 
            re_dev_dataset,
            lr=2e-5,
            epochs=5, 
            batch_size=batch_size
           )
  
  # 推断
  from ark_nlp.model.re.prgc_bert import Predictor
  
  prgc_re_predictor_instance = Predictor(model.module, tokenizer, re_train_dataset.cat2id)
  
  prgc_re_predictor_instance.predict_one_sample(待抽取文本)
  ```

## DisscussionGroup

- 公众号：**DataArk**

![wechat](https://github.com/xiangking/XK-PictureHost/blob/main/0.5Ark.jpg)

- wechat ID: **fk95624**
  
## Main contributors

<table>
<tr>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/xiangking>
            <img src=https://avatars.githubusercontent.com/u/29096754?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=xiangking/>
            <br />
            <sub style="font-size:14px"><b>xiangking</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/jimme0421>
            <img src=https://avatars.githubusercontent.com/u/43140191?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Jimme/>
            <br />
            <sub style="font-size:14px"><b>Jimme</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/Zrealshadow>
            <img src=https://avatars.githubusercontent.com/u/30857435?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Zrealshadow/>
            <br />
            <sub style="font-size:14px"><b>Zrealshadow</b></sub>
        </a>
    </td>
</tr>
</table>

## Acknowledge

  本项目用于收集和复现学术与工作中常用的NLP模型，整合成方便调用的形式，所以参考借鉴了网上很多开源实现，如有不当的地方，还请联系批评指教。
  在此，感谢大佬们的开源实现。
