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


import torch
import numpy as np

from ark_nlp.factory.predictor.base._predictor import Predictor


class PromptMLMPredictor(Predictor):

    def __init__(
        self,
        *args,
        prompt,
        prompt_mode='postfix',
        prefix_special_token_num=1,
        **kwargs
    ):
        super(PromptMLMPredictor, self).__init__(*args, **kwargs)
        self.prompt = prompt
        self.prompt_mode = prompt_mode
        self.prefix_special_token_num = prefix_special_token_num

    def _convert_to_transfomer_ids(
        self,
        text
    ):

        seq = self.tokenizer.tokenize(text)

        if self.prompt_mode == 'postfix':
            start_mask_position = len(seq) + self.prefix_special_token_num + self.prompt.index("[MASK]")
        else:
            start_mask_position = self.prefix_special_token_num + self.prompt.index("[MASK]")

        mask_position = [
            start_mask_position + index
            for index in range(len(self.tokenizer.tokenize(list(self.cat2id.keys())[0])))
        ]

        input_ids = self.tokenizer.sequence_to_ids(seq, self.prompt)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'mask_position': np.array(mask_position)
        }

        return features

    def predict_one_sample(
        self,
        text='',
        topk=1,
        return_label_name=True,
        return_proba=False
    ):
        if topk is None:
            topk = len(self.cat2id) if len(self.cat2id) > 2 else 1

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs).cpu().numpy()

        # [label_num, label_length]
        labels_ids = np.array(
            [self.tokenizer.vocab.convert_tokens_to_ids(
                self.tokenizer.tokenize(_cat)) for _cat in self.cat2id])

        preds = np.ones(shape=[len(labels_ids)])

        label_length = len(self.tokenizer.tokenize(list(self.cat2id.keys())[0]))

        for index in range(label_length):
            preds *= logit[index, labels_ids[:, index]]

        preds = torch.Tensor(preds)
        preds = preds.reshape(1, -1)

        probs, indices = preds.topk(topk, dim=1, sorted=True)

        preds = []
        probas = []
        for pred_, proba_ in zip(indices.cpu().numpy()[0], probs.cpu().numpy()[0].tolist()):

            if return_label_name:
                pred_ = self.id2cat[pred_]

            preds.append(pred_)

            if return_proba:
                probas.append(proba_)

        if return_proba:
            return list(zip(preds, probas))

        return preds
