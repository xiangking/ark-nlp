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

import copy
import numpy as np

from tqdm import tqdm
from ark_nlp.dataset.base._dataset import BaseDataset


class PURERCDataset(BaseDataset):
    """
    用于PURE bert联合关系分类任务的Dataset

    Args:
        data (DataFrame or string): 数据或者数据地址
        categories (list or None, optional): 数据类别, 默认值为: None
        is_retain_df (bool, optional): 是否将DataFrame格式的原始数据复制到属性retain_df中, 默认值为: False
        is_retain_dataset (bool, optional): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中, 默认值为: False
        is_train (bool, optional): 数据集是否为训练集数据, 默认值为: True
        is_test (bool, optional): 数据集是否为测试集数据, 默认值为: False
    """  # noqa: ignore flake8"

    def _get_categories(self):
        return sorted(
            list(
                set([triple_[4] for data_ in self.dataset
                     for triple_ in data_['triple']])) + ['None'])

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.strip())

        feature_names = list(data_df.columns)
        for index, row in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name: getattr(row, feature_name)
                for feature_name in feature_names
            })
        return dataset

    def _convert_to_transformer_ids(self, tokenizer):
        self.tokenizer = tokenizer

        if self.is_train:
            self.retain_dataset = copy.deepcopy(self.dataset)

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Converting sequence to transformer ids',
                )):
            text, entities = row['text'], row['entities']

            tokens = self.tokenizer.tokenize(text)

            index_token_mapping = self.tokenizer.get_token_mapping(text, tokens)

            start_mapping = {j[0]: i for i, j in enumerate(index_token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(index_token_mapping) if j}

            # 删去分词错误和超出最大长度的实体
            eidxs = []
            num = 0
            for eidx, entity in enumerate(entities):
                if entity[2] in start_mapping and entity[3] in end_mapping:
                    if self.tokenizer.max_seq_len - 3 - (num + 1) * 2 < end_mapping[
                            entity[3]]:
                        break
                    else:
                        eidxs.append(eidx)
                        num += 1

            entities_ = [entities[i] for i in eidxs]

            if len(tokens) > (self.tokenizer.max_seq_len - 3 - len(entities_) * 2):
                tokens = tokens[0:(self.tokenizer.max_seq_len - 3 - len(entities_) * 2)]

            # 生成position_ids
            position_ids = [i for i in range(len(tokens) + 2)]
            tokens += ['[SEP]']

            entity2marker = {}
            idx = len(tokens)
            for entity in entities_:
                head_idx, end_idx = entity[2], entity[3]
                tokens += [f'[{entity[1]}]']
                tokens += [f'[/{entity[1]}]']
                position_ids.append(start_mapping[head_idx] + 1)
                position_ids.append(end_mapping[end_idx] + 1)
                entity2marker[head_idx] = idx + 1
                entity2marker[end_idx] = idx + 2
                idx = idx + 2

            pad = [0] * (self.tokenizer.max_seq_len - len(position_ids))
            position_ids.extend(pad)

            input_ids, input_mask, segment_ids = self.tokenizer.sequence_to_ids(tokens)

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'position_ids': position_ids,
            }

            corres_tag = np.ones((self.tokenizer.max_seq_len,
                                  self.tokenizer.max_seq_len)) * self.cat2id['None']

            if not self.is_test:

                for triple in row['triples']:
                    sub_head_idx = triple[2]
                    sub_end_idx = triple[3]
                    obj_head_idx = triple[7]
                    obj_end_idx = triple[8]
                    if (sub_head_idx in entity2marker.keys()
                            and obj_head_idx in entity2marker.keys()
                            and sub_end_idx in entity2marker.keys()
                            and obj_end_idx in entity2marker.keys()):
                        corres_tag[entity2marker[sub_head_idx]][
                            entity2marker[obj_head_idx]] = self.cat2id[triple[4]]

            relations_idx = []
            entity_pair = []
            label_ids = []
            for eidx_sub, sub in zip(eidxs, entities_):
                for eidx_obj, obj in zip(eidxs, entities_):
                    if str(sub) == str(obj):
                        continue
                    sub_head_idx = entity2marker[sub[2]]
                    sub_end_idx = entity2marker[sub[3]]
                    obj_head_idx = entity2marker[obj[2]]
                    obj_end_idx = entity2marker[obj[3]]

                    relations_idx.append(
                        [sub_head_idx, sub_end_idx, obj_head_idx, obj_end_idx])
                    entity_pair.append([eidx_sub, eidx_obj])
                    label_ids.append(corres_tag[sub_head_idx][obj_head_idx])

            feature['relations_idx'] = relations_idx
            feature['entity_pair'] = entity_pair
            if not self.is_test:
                feature['label_ids'] = label_ids

            features.append(feature)

        return features
