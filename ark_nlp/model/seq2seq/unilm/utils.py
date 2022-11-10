import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data._utils.collate import default_collate


class Trie(object):
    """自定义Trie树对象，用来保存知识库
    """
    def __init__(self, value_key=-1):
        self.data = {}
        self.value_key = str(value_key)

    def __setitem__(self, key, value):
        """传入一对(key, value)到前缀树中
        """
        data = self.data
        for k in key:
            k = str(k)
            if k not in data:
                data[k] = {}
            data = data[k]
        if self.value_key in data:
            if data[self.value_key] != value:
                data[self.value_key] += ('\t' + value)
        else:
            data[self.value_key] = value

    def __getitem__(self, key):
        """获取key对应的value
        """
        data = self.data
        for k in key:
            k = str(k)
            data = data[k]
        return data[self.value_key]

    def next_ones(self, prefix):
        """获取prefix后一位的容许集
        """
        data = self.data
        for k in prefix:
            k = str(k)
            data = data[k]
        return [k for k in data if k != self.value_key]

    def keys(self, prefix=None, data=None):
        """获取以prefix开头的所有key
        """
        data = data or self.data
        prefix = prefix or []
        for k in prefix:
            k = str(k)
            if k not in data:
                return []
            data = data[k]
        results = []
        for k in data:
            if k == self.value_key:
                results.append([])
            else:
                results.extend([[k] + j for j in self.keys(None, data[k])])
        return [prefix + i for i in results]

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f, ensure_ascii=False)

    def load(self, filename):
        with open(filename) as f:
            self.data = json.load(f)


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


def take_along_dim(input_tensor, indices, dim=None):
    '''兼容部分低版本pytorch没有torch.take_along_dim
    '''
    if torch.__version__ >= '1.9.0':
        return torch.take_along_dim(input_tensor, indices, dim)
    else:
        # 该逻辑仅在少量数据上测试，如有bug，欢迎反馈
        if dim is None:
            res = input_tensor.flatten()[indices]
        else:
            res = np.take_along_axis(input_tensor.cpu().numpy(), indices.cpu().numpy(), axis=dim)
            res = torch.from_numpy(res).to(input_tensor.device)
        # assert res.equal(torch.take_along_dim(input_tensor, indices, dim))
        return res

# 转换知识库
# KG = Trie()
# if os.path.exists('../datasets/KG.json'):
#     KG.load('../datasets/KG.json')
# else:
#     with open('../datasets/Knowledge.txt') as f:
#         for l in tqdm(f):
#             s, p, o = l.split('\t')
#             s, m = subject_split(s)
#             ids = tokenizer.encode(s, p)[0][1:]
#             ids += tokenizer.encode(m)[0][1:-1]
#             KG[ids] = ' '.join(o.split())
#     KG.save('../datasets/KG.json')


class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略
    """
    def __init__(self, start_id, end_id, maxlen, minlen=1, device='cpu'):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.models = {}
        self.device = device
        if start_id is None:
            self.first_output_ids = torch.empty((1, 0), dtype=int, device=device)
        else:
            self.first_output_ids = torch.tensor([[self.start_id]], device=device)

    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数
        目前包含: 1. 设置rtype参数，并做相应处理；
                  2. 确定states的使用，并做相应处理；
                  3. 设置温度参数，并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(self, inputs, output_ids, states, temperature=1, rtype=default_rtype):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)

                if not use_states:
                    prediction = (prediction, None)

                if default_rtype == 'logits':
                    prediction = (nn.Softmax(dim=-1)(prediction[0] / temperature), prediction[1])
                elif temperature != 1:
                    probas = torch.power(prediction[0], 1.0 / temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == 'probas':
                    return prediction
                else:
                    return torch.log(prediction[0] + 1e-12), prediction[1]

            return new_predict

        return actual_decorator

    def predict(self, inputs, output_ids, states=None):
        """用户需自定义递归预测函数
        说明: 定义的时候，需要用wraps方法进行装饰，传入default_rtype和use_states，
             其中default_rtype为字符串logits或probas，probas时返回归一化的概率，
             rtype=logits时则返回softmax前的结果或者概率对数。
        返回: 二元组 (得分或概率, states)
        """
        raise NotImplementedError

    def beam_search(self, inputs_raw, topk, states=None, temperature=1, min_ends=1, add_btz_dim=True, **kwargs):
        """beam search解码
        说明: 这里的topk即beam size；
        返回: 最优解码序列。
        """
        inputs = []
        for i in inputs_raw:
            if isinstance(i, torch.torch.Tensor):
                pass
            elif isinstance(i, (list, tuple, np.ndarray)) and add_btz_dim:
                i = torch.tensor([i], device=self.device)
            elif isinstance(i, (list, tuple, np.ndarray)) and not add_btz_dim:
                i = torch.tensor(i, device=self.device)
            else:
                raise ValueError('Beam search inputs ele only support tensor、array、list、tuple')
            inputs.append(i)

        output_ids, output_scores = self.first_output_ids, torch.zeros(1, device=self.device)
        for step in range(self.maxlen):
            scores, states = self.predict(inputs, output_ids, states, temperature, 'logits')  # 计算当前得分
            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [i.repeat([topk]+[1]*(len(i.shape)-1)) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
            indices = scores.flatten().argsort(dim=-1, descending=True)[:topk]  # 仅保留topk
            if torch.__version__ <= '1.7.1':
                indices_1 = indices // scores.shape[1]  # 兼容老版本
            else:
                indices_1 = torch.div(indices, scores.shape[1], rounding_mode='floor')  # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = torch.cat([output_ids[indices_1], indices_2], 1)  # 更新输出
            output_scores = take_along_dim(scores, indices, dim=None)  # 更新得分
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best = output_scores.argmax()  # 得分最大的那个
                if is_end[best] and end_counts[best] >= min_ends:  # 如果已经终止
                    return output_ids[best]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        return output_ids[output_scores.argmax()]

    def random_sample(self, inputs, n, topk=None, topp=None, states=None, temperature=1, min_ends=1, **kwargs):
        """随机采样n个结果
        说明: 非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp
             表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        返回: n个解码序列组成的list。
        """
        inputs = [torch.tensor([i], device=self.device) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            probas, states = self.predict(inputs, output_ids, states, temperature, 'probas')  # 计算当前概率
            probas /= probas.sum(dim=-1, keepdims=True)  # 确保归一化
            if step == 0:  # 第1步预测后将结果重复n次
                probas = probas.repeat([n]+[1]*(len(probas.shape)-1))
                inputs = [i.repeat([n]+[1]*(len(i.shape)-1)) for i in inputs]
                output_ids = output_ids.repeat([n]+[1]*(len(output_ids.shape)-1))
            if topk is not None:
                k_indices = probas.argsort(dim=-1, descending=True)[:, :topk]  # 仅保留topk
                probas = take_along_dim(probas, k_indices, dim=1)  # topk概率
                probas /= probas.sum(dim=1, keepdims=True)  # 重新归一化
            if topp is not None:
                p_indices = probas.argsort(dim=-1, descending=True)  # 从高到低排序
                probas = take_along_dim(probas, p_indices, dim=-1)  # 排序概率
                cumsum_probas = torch.cumsum(probas, dim=-1)  # 累积概率
                flag = torch.roll(cumsum_probas >= topp, 1, dims=1)  # 标记超过topp的部分
                flag[:, 0] = False  # 结合上面的torch.roll，实现平移一位的效果
                probas[flag] = 0  # 后面的全部置零
                probas /= probas.sum(dim=1, keepdims=True)  # 重新归一化

            sample_func = lambda p: torch.multinomial(p, 1)  # 按概率采样函数
            sample_ids = torch.stack([sample_func(p) for p in probas])
            sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
            if topp is not None:
                sample_ids = take_along_dim(p_indices, sample_ids, dim=1)  # 对齐原id
            if topk is not None:
                sample_ids = take_along_dim(k_indices, sample_ids, dim=1)  # 对齐原id
            output_ids = torch.cat([output_ids, sample_ids], 1)  # 更新输出
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                flag = is_end & (end_counts >= min_ends)  # 标记已完成序列
                if flag.any():  # 如果有已完成的
                    for ids in output_ids[flag]:  # 存好已完成序列
                        results.append(ids)
                    flag = (flag == False)  # 标记未完成序列
                    inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
                    output_ids = output_ids[flag]  # 只保留未完成部分候选集
                    end_counts = end_counts[flag]  # 只保留未完成部分end计数
                    if len(output_ids) == 0:
                        break
        # 如果还有未完成序列，直接放入结果
        for ids in output_ids:
            results.append(ids)
        # 返回结果
        return results


class AutoQA(AutoRegressiveDecoder):
    """seq2seq解码器
    """
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

        y_pred = self.model(input_ids=token_ids, token_type_ids=segment_ids)

        if self.KG is None:
            return F.softmax(y_pred[:, index_, :], dim=-1)
        else:
            probas = F.softmax(y_pred[:, index_, :], dim=-1)
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

    def generate(self, inputs, tokenizer, module=None, KG=None, **kwargs):
        self.model = module
        self.tokenizer = tokenizer
        self.KG = KG

        token_ids, segment_ids = inputs['text_ids'], inputs['text_token_type_ids']

        subject_ids_list = []

        for num in range(len(token_ids)):

            output_ids = self.beam_search([token_ids[num], segment_ids[num]], **kwargs)  # 基于beam search
            # end_idxs = [i for i, j in enumerate(output_ids) if j == self.end_id]
            #
            # if len(end_idxs) > 0:
            #     subject_ids = output_ids[:end_idxs[0]]
            # else:
            #     subject_ids = []

            subject_ids_list.append(''.join(tokenizer.vocab.convert_ids_to_tokens(output_ids[:-1])))

        return subject_ids_list
