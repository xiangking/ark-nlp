# Copyright (c) 2022 DataArk Authors. All Rights Reserved.
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

import numpy as np

from tqdm import tqdm
from ark_nlp.dataset import SentenceClassificationDataset


class PromptDataset(SentenceClassificationDataset):
    """
    用于使用prompt的自然语言处理任务的Dataset

    Args:
        data (DataFrame or string): 数据或者数据地址
        prompt (list): 所使用的prompt, 如["是", "[MASK]"]
        prompt_mode (string): 
            prompt放置在文本中的方式
            有postfix和prefix两种, postfix表示text + prompt, prefix表示prompt + text
            默认值为"postfix"
        prefix_special_token_num (int): 使用前缀特殊字符的数量, 如"[CLS]", 默认值为1
        categories (list, optional): 数据类别, 默认值为None
        do_retain_df (bool, optional): 是否将DataFrame格式的原始数据复制到属性retain_df中, 默认值为: False
        do_retain_dataset (bool, optional): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中, 默认值为: False
        is_train (bool, optional): 数据集是否为训练集数据, 默认值为: True
        is_test (bool, optional): 数据集是否为测试集数据, 默认值为: False
        progress_verbose (bool, optional): 是否显示数据进度, 默认值为: True
    """  # noqa: ignore flake8"

    def __init__(self,
                 *args,
                 prompt,
                 prompt_mode='postfix',
                 prefix_special_token_num=1,
                 **kwargs):
        super(PromptDataset, self).__init__(*args, **kwargs)

        self.prompt = prompt
        self.mask_lm_label_size = self.prompt.count("[MASK]")

        self.prompt_mode = prompt_mode
        self.prefix_special_token_num = prefix_special_token_num

    def _convert_to_transformer_ids(self, tokenizer):

        features = []

        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Convert to transformer ids',
                )):

            seq = tokenizer.tokenize(row['text'])

            if self.prompt_mode == 'postfix':
                start_mask_position = len(seq) + self.prefix_special_token_num + self.prompt.index("[MASK]")
            else:
                start_mask_position = self.prefix_special_token_num + self.prompt.index("[MASK]")

            mask_position = [
                start_mask_position + index for index in range(self.mask_lm_label_size)
            ]

            input_ids = tokenizer.sequence_to_ids(row['text'], self.prompt,
                                                  self.prompt_mode)

            input_ids, attention_mask, token_type_ids = input_ids

            feature = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'mask_position': np.array(mask_position, dtype='int64')
            }

            if not self.is_test:
                mask_lm_label = tokenizer.vocab.convert_tokens_to_ids(
                    tokenizer.tokenize(row['label']))

                feature['label_ids'] = np.array(mask_lm_label, dtype='int64')

            features.append(feature)

        return features
