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
import numpy as np

from ark_nlp.dataset import TokenClassificationDataset


class W2NERDataset(TokenClassificationDataset):
    """
    W2NER的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def _get_categories(self):
        categories = ['<none>', '<suc>'] + sorted(list(set([label_['type'] for data in self.dataset for label_ in data['label']])))
        return categories

    @staticmethod
    def convert_index_to_text(index, type):
        text = "-".join([str(i) for i in index])
        text = text + "-#-{}".format(type)
        return text

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []

        for (index_, row_) in enumerate(self.dataset):

            tokens = bert_tokenizer.tokenize(row_['text'])[:bert_tokenizer.max_seq_len-2]

            input_ids = bert_tokenizer.sequence_to_ids(tokens)
            input_ids, input_mask, segment_ids = input_ids

            # input_length 对应源码 sent_length
            input_length = len(tokens)
            _grid_mask2d = np.ones((input_length, input_length), dtype=np.bool)
            _dist_inputs = np.zeros((input_length, input_length), dtype=np.int)
            _pieces2word = np.zeros((input_length, input_length+2), dtype=np.bool)

            # pieces2word 类似于token_mapping
            start = 0
            for i, pieces in enumerate(tokens):
                # 对齐源码
                pieces = [pieces]
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

            # dist_inputs
            # https://github.com/ljynlp/W2NER/issues/17
            dis2idx = np.zeros((1000), dtype='int64')
            dis2idx[1] = 1
            dis2idx[2:] = 2
            dis2idx[4:] = 3
            dis2idx[8:] = 4
            dis2idx[16:] = 5
            dis2idx[32:] = 6
            dis2idx[64:] = 7
            dis2idx[128:] = 8
            dis2idx[256:] = 9

            for k in range(input_length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i in range(input_length):
                for j in range(input_length):
                    if _dist_inputs[i, j] < 0:
                        _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                    else:
                        _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
            _dist_inputs[_dist_inputs == 0] = 19

            _grid_labels = np.zeros((
                bert_tokenizer.max_seq_len,
                bert_tokenizer.max_seq_len,)
            )

            for info_ in row_["label"]:
                index = info_['idx']

                if len(index) > 0 and index[-1] < bert_tokenizer.max_seq_len:

                    for i in range(len(index)):
                        if i + 1 >= len(index):
                            break
                        _grid_labels[index[i], index[i + 1]] = self.cat2id['<suc>']
                    _grid_labels[index[-1], index[0]] = self.cat2id[info_["type"]]

            # 源码中 collate_fn 中处理成 max_lenth * max_lenth 矩阵代码
            def fill(data, new_data):
                new_data[:data.shape[0], :data.shape[1]] = torch.tensor(data, dtype=torch.long)
                return new_data

            mask2d_mat = torch.zeros((bert_tokenizer.max_seq_len, bert_tokenizer.max_seq_len), dtype=torch.long)
            _grid_mask2d = fill(_grid_mask2d, mask2d_mat)
            dis_mat = torch.zeros((bert_tokenizer.max_seq_len, bert_tokenizer.max_seq_len), dtype=torch.long)
            _dist_inputs = fill(_dist_inputs, dis_mat)
            sub_mat = torch.zeros((bert_tokenizer.max_seq_len, bert_tokenizer.max_seq_len), dtype=torch.long)
            _pieces2word = fill(_pieces2word, sub_mat)
            labels_mat = torch.zeros((bert_tokenizer.max_seq_len, bert_tokenizer.max_seq_len), dtype=torch.long)
            _grid_labels = fill(_grid_labels, labels_mat)

            _entity_text = list(set([W2NERDataset.convert_index_to_text(info_['idx'],
                                self.cat2id[info_["type"]]) for info_ in row_['label']]))

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'grid_mask2d': _grid_mask2d,
                'dist_inputs': _dist_inputs,
                'pieces2word': _pieces2word,
                'label_ids': _grid_labels,
                'input_lengths': input_length,
                'entity_text': _entity_text,
            }

            features.append(feature)

        return features
