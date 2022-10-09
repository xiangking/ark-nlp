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
# Author: Xiang Wang, xiangking1995@163.com
# Status: Active

import warnings
import unicodedata
import transformers
import numpy as np

from typing import List
from ark_nlp.processor.tokenizer._tokenizer import BaseTokenizer
from ark_nlp.processor.tokenizer.tokenizer_utils import Trie
from ark_nlp.processor.tokenizer.tokenizer_utils import BasicTokenizer
from ark_nlp.processor.tokenizer.tokenizer_utils import WordpieceTokenizer


class TransformerTokenizer(BaseTokenizer):
    """
    Transformer文本编码器, 用于对文本进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名, 用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
    """  # noqa: ignore flake8"

    def __init__(
        self,
        vocab,
        max_seq_len,
        *,
        do_lower_case=None,
        never_split=None,
        unk_token=None,
        sep_token=None,
        pad_token=None,
        cls_token=None,
        mask_token=None,
        bos_token=None,
        eos_token=None,
        space_token=None,
        additional_special_tokens=None,
        strip_accents=None,
    ):

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.space_token = space_token
        self.additional_special_tokens = additional_special_tokens

        if isinstance(vocab, str):
            # TODO: 改成由自定义的字典所决定
            vocab = transformers.AutoTokenizer.from_pretrained(vocab)

        self.vocab = vocab
        self.init_vocab_special_tokens()

        if self.additional_special_tokens is None:
            self.additional_special_tokens = set()

        self.max_seq_len = max_seq_len
        self.tokenizer_type = 'transformer'

        if do_lower_case is None:
            self.do_lower_case = getattr(self.vocab, 'do_lower_case', True)
        else:
            self.do_lower_case = do_lower_case

        self.strip_accents = strip_accents

        self.never_split = set(self.vocab.all_special_tokens)
        if never_split:
            self.never_split |= never_split

        # trie树主要是为了special_tokens的分词
        self.tokens_trie = self._create_trie(self.never_split)

        self.basic_tokenizer = BasicTokenizer(do_lower_case=self.do_lower_case,
                                              strip_accents=self.strip_accents)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab.vocab,
                                                      unk_token=self.vocab.unk_token)

        self.token_dict = self.vocab.vocab

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            trie.add(token)
        return trie

    def tokenize(
        self,
        text,
        do_basic_tokenize=True,
        use_token_dict_in_basic_tokenizer=True,
        do_keep_space_space_token=False,
        use_unk_token=True,
    ):

        if do_keep_space_space_token:
            if self.space_token is None:
                raise ValueError("Please set space token")

            text = "".join([char if char != " " else self.space_token for char in text])

        text_pieces = self.tokens_trie.split(text)
        split_tokens = []

        for text_piece in text_pieces:
            if not text_piece:
                continue
            elif text_piece in self.token_dict:
                split_tokens.append(text_piece)
            elif do_basic_tokenize:
                for token in self.basic_tokenizer.tokenize(
                        text_piece,
                        token_dict=self.token_dict
                        if use_token_dict_in_basic_tokenizer else None):
                    for sub_token in self.wordpiece_tokenizer.tokenize(
                            token, use_unk_token):
                        split_tokens.append(sub_token)
            else:
                split_tokens.extend(self.wordpiece_tokenizer.tokenize(text_piece))

        return split_tokens

    def init_vocab_special_tokens(self):

        if self.unk_token:
            self.vocab.add_special_tokens({'unk_token': self.unk_token})
        else:
            self.unk_token = getattr(self.vocab, '_unk_token', None)

        if self.sep_token:
            self.vocab.add_special_tokens({'sep_token': self.sep_token})
        else:
            self.sep_token = getattr(self.vocab, '_sep_token', None)

        if self.pad_token:
            self.vocab.add_special_tokens({'pad_token': self.pad_token})
        else:
            self.pad_token = getattr(self.vocab, '_pad_token', None)

        if self.cls_token:
            self.vocab.add_special_tokens({'cls_token': self.cls_token})
        else:
            self.cls_token = getattr(self.vocab, '_cls_token', None)

        if self.mask_token:
            self.vocab.add_special_tokens({'mask_token': self.mask_token})
        else:
            self.mask_token = getattr(self.vocab, '_mask_token', None)

        if self.bos_token:
            self.vocab.add_special_tokens({'bos_token': self.bos_token})
        else:
            self.bos_token = getattr(self.vocab, '_bos_token', None)

        if self.eos_token:
            self.vocab.add_special_tokens({'eos_token': self.eos_token})
        else:
            self.eos_token = getattr(self.vocab, '_eos_token', None)

        if self.space_token:

            if self.space_token not in self.vocab.vocab:
                warnings.warn(
                    f"space_token='{self.space_token}'并不在预训练模型的词典。"
                    "因此, 请为其在预训练模型中增加词典嵌入module.resize_token_embeddings(vocab_size)或更换space_token。"
                    "特别注意的是'ernie-1.0'预先预留了部分嵌入位置, 因此不需要resize_token_embeddings。")

            self.vocab.add_special_tokens(
                {'additional_special_tokens': [self.space_token]})

        if self.additional_special_tokens:
            self.vocab.add_special_tokens(
                {'additional_special_tokens': self.additional_special_tokens})

    def add_special_tokens(self, special_tokens) -> int:

        if type(special_tokens) == List:
            special_tokens = {'additional_special_tokens': special_tokens}

        added_tokens = self.vocab.add_special_tokens(special_tokens)
        self.tokens_trie = self._create_trie(self.never_split
                                             | set(self.all_special_tokens))
        self.token_dict = self.vocab.vocab

        return added_tokens

    @property
    def all_special_tokens(self):
        return self.vocab.all_special_tokens

    def get_token_mapping(
        self,
        text,
        tokens,
        *,
        is_mapping_index=True,
        do_basic_tokenize=True,
        use_token_dict_in_basic_tokenizer=True,
        do_keep_space_space_token=False,
        use_unk_token=False,
        never_skip_tokens=None,
    ):
        """给出原始的text和tokenize后的tokens的映射关系"""
        token_size = len(tokens)
        if self.do_lower_case:
            text = text.lower()
            if never_skip_tokens:
                never_skip_tokens = [token.lower() for token in never_skip_tokens]

        # 针对分词后出现连续unk token从而无法判断单个unk token占位大小的问题
        # 通过进行不使用unk token的分词获取tokens从而获取token_mapping
        if self.unk_token and (self.unk_token * 2) in "".join(tokens):
            tokens = self.tokenize(
                text,
                do_basic_tokenize=do_basic_tokenize,
                use_token_dict_in_basic_tokenizer=use_token_dict_in_basic_tokenizer,
                do_keep_space_space_token=do_keep_space_space_token,
                use_unk_token=use_unk_token)
            tokens = tokens[:token_size]

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self.do_lower_case and self.strip_accents is not False:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token_index, token in enumerate(tokens):
            if self.do_lower_case:
                token = token.lower()

            while text[offset] == ' ' and self.space_token is None:
                offset += 1

            # 针对space token有意义的情况
            if self.space_token and (token == self.space_token
                                     or token == self.space_token.lower()):
                token_mapping.append(
                    char_mapping[offset:offset +
                                 1] if is_mapping_index else text[offset:offset + 1])
                offset = offset + 1
            # 针对单个unk token, 使用下一个token的位置来获取unk token对应的占位大小
            elif self.unk_token and (token == self.unk_token
                                     or token == self.unk_token.lower()):
                # 如果当前的unk token已经在句尾, 则可知其占位大小为1
                # 否则, 通过获取其下一个token的位置进行offset的计算
                if token_index == len(tokens) - 1:
                    token_mapping.append(
                        char_mapping[offset:offset +
                                     1] if is_mapping_index else text[offset:offset + 1])
                    offset = offset + 1
                else:
                    next_token = self.recover_bert_token(tokens[token_index + 1])

                    # 若下一个token是空格, 则使用空格来检索, 否则正常进行token检索
                    if (self.space_token and (next_token == self.space_token
                                              or next_token == self.space_token.lower())):
                        next_token_start_idx = text[offset:].index(' ') + offset
                    else:
                        next_token_start_idx = text[offset:].index(next_token) + offset

                    # 判断unk与下一个token直接是否存在空格，若存在, 同样使用空格检索
                    if text[next_token_start_idx - 1] == ' ':
                        next_token_start_idx = text[offset:].index(' ') + offset

                    token_mapping.append(
                        char_mapping[offset:next_token_start_idx]
                        if is_mapping_index else text[offset:next_token_start_idx])
                    offset = next_token_start_idx
            # 针对用户自定义的不能跳过的特殊符号
            elif never_skip_tokens and token in never_skip_tokens:
                token_mapping.append(
                    char_mapping[offset:offset +
                                 1] if is_mapping_index else text[offset:offset + 1])
                offset = offset + 1
            # 针对[CLS]或者是[SEP]之类的特殊词, 没有对应的映射
            # PS: 由于ark-nlp的分词不会添加特殊词, 因此理论上不会触发该条件
            elif self._is_special(token):
                token_mapping.append([])
                start_idx = text[offset:].index(token) + offset
                end_idx = start_idx + len(token)
                offset = end_idx
            else:
                token = self.recover_bert_token(token)
                start_idx = text[offset:].index(token) + offset
                end_idx = start_idx + len(token)
                token_mapping.append(char_mapping[start_idx:end_idx]
                                     if is_mapping_index else text[start_idx:end_idx])
                offset = end_idx

        return token_mapping

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def recover_bert_token(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def sequence_to_ids(self, sequence_a, sequence_b=None, **kwargs):
        if sequence_b is None:
            return self.sentence_to_ids(sequence_a, **kwargs)
        else:
            return self.pair_to_ids(sequence_a, sequence_b, **kwargs)

    def sentence_to_ids(self, sequence, return_sequence_length=False):
        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        if return_sequence_length:
            sequence_length = len(sequence)

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 2:
            sequence = sequence[0:(self.max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence + ['[SEP]']
        segment_ids = [0] * len(sequence)
        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        if return_sequence_length:
            return (sequence, sequence_mask, segment_ids, sequence_length)

        return (sequence, sequence_mask, segment_ids)

    def pair_to_ids(self,
                    sequence_a,
                    sequence_b,
                    return_sequence_length=False,
                    truncation_method='average'):
        if type(sequence_a) == str:
            sequence_a = self.tokenize(sequence_a)

        if type(sequence_b) == str:
            sequence_b = self.tokenize(sequence_b)

        if return_sequence_length:
            sequence_length = (len(sequence_a), len(sequence_b))

        # 对超长序列进行截断
        if truncation_method == 'average':
            if len(sequence_a) > ((self.max_seq_len - 3) // 2):
                sequence_a = sequence_a[0:(self.max_seq_len - 3) // 2]
            if len(sequence_b) > ((self.max_seq_len - 3) // 2):
                sequence_b = sequence_b[0:(self.max_seq_len - 3) // 2]
        elif truncation_method == 'last':
            if len(sequence_b) > (self.max_seq_len - 3 - len(sequence_a)):
                sequence_b = sequence_b[0:(self.max_seq_len - 3 - len(sequence_a))]
        elif truncation_method == 'first':
            if len(sequence_a) > (self.max_seq_len - 3 - len(sequence_b)):
                sequence_a = sequence_a[0:(self.max_seq_len - 3 - len(sequence_b))]
        else:
            raise ValueError("The truncation method does not exist")

        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence_a + ['[SEP]'] + sequence_b + ['[SEP]']
        segment_ids = [0] * (len(sequence_a) + 2) + [1] * (len(sequence_b) + 1)

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        if return_sequence_length:
            return (sequence, sequence_mask, segment_ids, sequence_length)

        return (sequence, sequence_mask, segment_ids)


class SentenceTokenizer(TransformerTokenizer):
    """
    Transformer文本编码器, 用于单句子进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名, 用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
    """  # noqa: ignore flake8"

    def sequence_to_ids(self, sequence, **kwargs):
        return self.sentence_to_ids(sequence, **kwargs)


class PairTokenizer(TransformerTokenizer):
    """
    Transformer文本编码器, 用于句子对拼接进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名, 用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
    """  # noqa: ignore flake8"

    def sequence_to_ids(self, sequence_a, sequence_b, **kwargs):
        return self.pair_to_ids(sequence_a, sequence_b, **kwargs)


class TokenTokenizer(TransformerTokenizer):
    """
    Transformer文本编码器, 用于按字符进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名, 用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
    """  # noqa: ignore flake8"

    def tokenize(self, text, **kwargs):
        tokens = []
        for token in text:
            if token == ' ':
                tokens.extend([token])
            tokens.extend(self.vocab.tokenize(token))
        return tokens

    def sequence_to_ids(self, sequence, **kwargs):
        return self.sentence_to_ids(sequence, **kwargs)


class SpaceTokenizer(TransformerTokenizer):
    """
    Transformer文本编码器, 用于对文本(基于分隔符分割维度)进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名, 用于实现文本分词和ID化
        max_seq_len (int): 
            预设的文本最大长度
        split_token (string):
            分隔符
        additional_special_split_token (string):
            额外添加的特殊字符
    """  # noqa: ignore flake8"

    def __init__(self, vocab, max_seq_len, space_token='[unused1]', **kwargs):
        super(SpaceTokenizer, self).__init__(vocab,
                                             max_seq_len,
                                             space_token=space_token,
                                             **kwargs)

    def tokenize(self, text, do_keep_space_space_token=True, **kwargs):
        return super().tokenize(text,
                                do_keep_space_space_token=do_keep_space_space_token,
                                **kwargs)

    def get_token_mapping(self, text, tokens, do_keep_space_space_token=True, **kwargs):
        return super().get_token_mapping(
            text, tokens, do_keep_space_space_token=do_keep_space_space_token, **kwargs)

    def sequence_to_ids(self, sequence, **kwargs):
        return self.sentence_to_ids(sequence, **kwargs)


class PromptMLMTransformerTokenizer(TransformerTokenizer):
    """
    模板学习Transformer文本编码器, 用于对文本进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名, 用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
    """
    def sequence_to_ids(self,
                        sequence,
                        prompt,
                        prompt_mode='postfix',
                        return_sequence_length=False,
                        **kwargs):
        """
        将序列ID化

        Args:
            sequence (string or list): 输入序列
            prompt (list): 所使用的prompt, 如["是", "[MASK]"]
            prompt_mode (string):
                prompt放置在文本中的方式
                有postfix和prefix两种, postfix表示text + prompt, prefix表示prompt + text
                默认值为"postfix"
            return_sequence_length (bool, optional): 返回是否包含序列长度, 默认值为False
        """  # noqa: ignore flake8"

        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        if return_sequence_length:
            sequence_length = len(sequence)

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 2 - len(prompt):
            sequence = sequence[0:(self.max_seq_len - 2 - len(prompt))]

        # 分别在首尾拼接特殊符号
        if prompt_mode == 'postfix':
            sequence = ['[CLS]'] + sequence + prompt + ['[SEP]']
        else:
            sequence = ['[CLS]'] + prompt + sequence + ['[SEP]']

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        segment_ids = [0] * len(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        if return_sequence_length:
            return (sequence, sequence_mask, segment_ids, sequence_length)

        return (sequence, sequence_mask, segment_ids)
