# Copyright (c) 2020 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Chenjie Shen, jimme.shen123@gmail.com
# Status: Active

import torch
import torch.nn.functional as F
from collections import defaultdict

from ark_nlp.model.seq2seq.unilm.utils import AutoRegressiveDecoder


def lcs(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]


class UniLMPredictor(AutoRegressiveDecoder):
    """
    UniLM bert模型的生成器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        start_id: 开始id
        end_id: 结束的id
        maxlen: 生成的文本最大长度
        minlen=1: 生成的文本最小长度
        KG=None: 生成结果token的Trie型的字典树，参考
        device='cpu'
        
    """  # noqa: ignore flake8"
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=self.device)], 1)

        # 根据max_seq_len与seq的长度产生填充序列
        index_ = token_ids.shape[1]
        padding = [0] * (self.tokenizer.max_seq_len - index_)
        padding = torch.tensor(padding, device=self.device).unsqueeze(0)
        padding = padding.repeat(token_ids.shape[0], 1)

        token_ids = torch.cat([token_ids, padding], 1)
        segment_ids = torch.cat([segment_ids, padding], 1)

        y_pred = self.module(input_ids=token_ids, token_type_ids=segment_ids)

        if self.KG is None:
            return F.softmax(y_pred[:, index_-1, :], dim=-1)
        else:
            probas = F.softmax(y_pred[:, index_-1, :], dim=-1)
            new_probas = torch.zeros_like(probas)
            for i, ids in enumerate(output_ids):
                ids = ids.cpu().numpy()
                next_ids = [int(j) for j in self.KG.next_ones(ids)]  # 下一位容许集
                # ===========如果t时刻为Pt的前缀树中的短句，带来的信息增益越大，则增加Pt的概率
                if len(next_ids) > 1 and self.end_id in ids:  # 容许集大于1且已解码出S
                    candidates = self.KG.keys(list(ids))  # 可能解码结果
                    weights = torch.ones_like(probas[i])  # 默认权重为1
                    lcs0 = lcs(ids, token_ids[i])[0]  # 当前已经覆盖的token数
                    for c in candidates:
                        if len(c) > len(ids):
                            c = [int(j) for j in c]
                            w = lcs(c, token_ids[i])[0] - lcs0  # 未来还可能覆盖的token数
                            weights[c[len(ids)]] = max(w + 1, weights[c[len(ids)]].cpu().numpy())
                    probas[i] = torch.pow(probas[i], 1. / weights)  # 按 p^(1/n) 来增大权重
                if not next_ids:  # 如果容许集为空，意味着要结束了
                    next_ids.append(self.end_id)
                new_probas[i, next_ids] += probas[i, next_ids]  # 只保留容许集概率
            new_probas /= new_probas.sum(axis=1, keepdims=True)  # 重新归一化
            return new_probas

    def _convert_to_transformer_ids(self, text):

        text_tokens = self.tokenizer.tokenize(text)

        if len(text_tokens) > self.tokenizer.max_seq_len - 3 - self.maxlen:
            text_tokens = text_tokens[:self.tokenizer.max_seq_len - 3 - self.maxlen]

        token_ids = self.tokenizer.vocab.convert_tokens_to_ids(['[CLS]'] + text_tokens + ['[SEP]'])
        segment_ids = [0] * len(token_ids)

        feature = {
            'input_ids': token_ids,
            'token_type_ids': segment_ids,
        }

        return feature

    def _get_input_ids(self,
                       text,
                       ):
        if self.tokenizer.tokenizer_type == 'transformer':
            return self._convert_to_transformer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def predict_one_sample(self, text='', **kwargs):

        """
        单样本预测

        Args:
            text (string): 输入文本
            topk (int, optional): 返回TopK结果, 默认值为: None
            return_label_name (bool, optional): 返回结果的标签ID转化成原始标签, 默认值为: True
            return_proba (bool, optional): 返回结果是否带上预测的概率, 默认值为: False
        """  # noqa: ignore flake8"

        features = self._get_input_ids(text)

        inputs = self._get_module_one_sample_inputs(features)

        output_ids = self.beam_search([inputs['input_ids'], inputs['token_type_ids']], **kwargs)  # 基于beam search

        return ''.join(self.tokenizer.vocab.convert_ids_to_tokens(output_ids[:-1]))