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

from tqdm import tqdm
from ark_nlp.dataset import TokenClassificationDataset


class W2NERDataset(TokenClassificationDataset):
    """
    W2NER的Dataset

    Args:
        data (DataFrame or string): 数据或者数据地址
        categories (list or None, optional): 数据类别, 默认值为: None
        do_retain_df (bool, optional): 是否将DataFrame格式的原始数据复制到属性retain_df中, 默认值为: False
        do_retain_dataset (bool, optional): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中, 默认值为: False
        is_train (bool, optional): 数据集是否为训练集数据, 默认值为: True
        is_test (bool, optional): 数据集是否为测试集数据, 默认值为: False
        progress_verbose (bool, optional): 是否显示数据进度, 默认值为: True
    """  # noqa: ignore flake8"

    def _get_categories(self):
        categories = ['<none>', '<suc>'] + sorted(
            list(
                set([label_['type'] for data in self.dataset
                     for label_ in data['label']])))
        return categories

    @staticmethod
    def convert_index_to_text(index, type):
        text = "-".join([str(i) for i in index])
        text = text + "-#-{}".format(type)
        return text

    def _convert_to_transformer_ids(self, tokenizer):

        features = []

        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Convert to transformer ids',
                )):
            tokens = tokenizer.tokenize(row['text'])[:tokenizer.max_seq_len - 2]

            input_ids = tokenizer.sequence_to_ids(tokens)
            input_ids, attention_mask, token_type_ids = input_ids

            # sequence_length 对应源码 sent_length
            sequence_length = len(tokens)
            grid_mask2d = np.ones((sequence_length, sequence_length), dtype=np.bool)
            dist_inputs = np.zeros((sequence_length, sequence_length), dtype=np.int)
            pieces2word = np.zeros((sequence_length, sequence_length + 2), dtype=np.bool)

            # pieces2word 类似于token_mapping
            start = 0
            for i, pieces in enumerate(tokens):
                # 对齐源码
                pieces = [pieces]
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
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

            for k in range(sequence_length):
                dist_inputs[k, :] += k
                dist_inputs[:, k] -= k

            for i in range(sequence_length):
                for j in range(sequence_length):
                    if dist_inputs[i, j] < 0:
                        dist_inputs[i, j] = dis2idx[-dist_inputs[i, j]] + 9
                    else:
                        dist_inputs[i, j] = dis2idx[dist_inputs[i, j]]
            dist_inputs[dist_inputs == 0] = 19

            grid_labels = np.zeros((
                tokenizer.max_seq_len,
                tokenizer.max_seq_len,
            ))

            for info_ in row["label"]:
                index = info_['idx']

                if len(index) > 0 and index[-1] < tokenizer.max_seq_len:

                    for i in range(len(index)):
                        if i + 1 >= len(index):
                            break
                        grid_labels[index[i], index[i + 1]] = self.cat2id['<suc>']
                    grid_labels[index[-1], index[0]] = self.cat2id[info_["type"]]

            # 源码中 collate_fn 中处理成 max_lenth * max_lenth 矩阵代码
            def fill(data, new_data):
                new_data[:data.shape[0], :data.shape[1]] = torch.tensor(data,
                                                                        dtype=torch.long)
                return new_data

            mask2d_mat = torch.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len),
                                     dtype=torch.long)
            grid_mask2d = fill(grid_mask2d, mask2d_mat)
            dis_mat = torch.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len),
                                  dtype=torch.long)
            dist_inputs = fill(dist_inputs, dis_mat)
            sub_mat = torch.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len),
                                  dtype=torch.long)
            pieces2word = fill(pieces2word, sub_mat)
            labels_mat = torch.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len),
                                     dtype=torch.long)
            grid_labels = fill(grid_labels, labels_mat)

            entity_text = list(
                set([
                    W2NERDataset.convert_index_to_text(info_['idx'],
                                                       self.cat2id[info_["type"]])
                    for info_ in row['label']
                ]))

            feature = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'grid_mask2d': grid_mask2d,
                'dist_inputs': dist_inputs,
                'pieces2word': pieces2word,
                'label_ids': grid_labels,
                'sequence_length': sequence_length,
                'entity_text': entity_text,
            }

            features.append(feature)

        return features
