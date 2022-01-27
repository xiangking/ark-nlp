# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
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
# Author: Xiang Wang, xiangking1995@163.com
# Status: Active


from ark_nlp.dataset import TwinTowersSentenceClassificationDataset


class UnsupervisedSimCSEDataset(TwinTowersSentenceClassificationDataset):
    """
    用于无监督的SimCSE模型文本匹配任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def _get_categories(self):
        if 'label' in self.dataset[0]:
            return sorted(list(set([data['label'] for data in self.dataset])))
        else:
            return None

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (_index, _row) in enumerate(self.dataset):

            input_ids_a = bert_tokenizer.sequence_to_ids(_row['text_a'])
            input_ids_b = bert_tokenizer.sequence_to_ids(_row['text_b'])

            input_ids_a, input_mask_a, segment_ids_a = input_ids_a
            input_ids_b, input_mask_b, segment_ids_b = input_ids_b

            feature = {
                'input_ids_a': input_ids_a,
                'attention_mask_a': input_mask_a,
                'token_type_ids_a': segment_ids_a,
                'input_ids_b': input_ids_b,
                'attention_mask_b': input_mask_b,
                'token_type_ids_b': segment_ids_b
            }

            features.append(feature)

            if 'label' in _row:
                label_ids = self.cat2id[_row['label']]
                feature['label_ids'] = label_ids

        return features
