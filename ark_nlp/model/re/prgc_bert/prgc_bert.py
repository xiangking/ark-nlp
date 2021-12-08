import torch

from torch import nn
from transformers import BertModel
from transformers import BertPreTrainedModel
from collections import Counter


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class SequenceLabelForSO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output


class PRGCBert(BertPreTrainedModel):
    """
    PRGC Bert命名实体模型

    Args:
        config: 模型的配置对象
        seq_tag_size (:obj:`int`, optional, defaults to 3): 序列标注子任务的字符标签个数
        drop_prob (:obj:`float`, optional, defaults to 0.3): dropout rate，bert之外的模型统一使用这个数值的dropout
        emb_fusion (:obj:`string`, optional, defaults to `concat`): 关系嵌入与bert输出向量的融合方式，concat是拼接，sum是加和
        corres_mode (:obj:`string` or :obj:`string`, optional, defaults to None): 生成global correspondence矩阵的方式，
                                                                                  biaffine是使用biaffine交叉主体和客体向量进行生成，比较节约显存，
                                                                                  None则是原论文方式，通过拼接向量再使用全连接层生成
        biaffine_hidden_size (:obj:`int`, optional, defaults to 128): 若使用biaffine生成global correspondence矩阵时，biaffine的隐层size

    Reference:
        [1] PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction
        [2] https://github.com/hy-struggle/PRGC
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        seq_tag_size=3,
        drop_prob=0.3,
        emb_fusion='concat',
        corres_mode=None,
        biaffine_hidden_size=128,
    ):
        super().__init__(config)
        self.seq_tag_size = seq_tag_size
        self.rel_num = config.num_labels
        self.emb_fusion = emb_fusion

        # pretrain model
        self.bert = BertModel(config)
        # sequence tagging
        self.sequence_tagging_sub = MultiNonLinearClassifier(
            config.hidden_size * 2,
            self.seq_tag_size, drop_prob
        )
        self.sequence_tagging_obj = MultiNonLinearClassifier(
            config.hidden_size * 2,
            self.seq_tag_size,
            drop_prob
        )
        self.sequence_tagging_sum = SequenceLabelForSO(
            config.hidden_size,
            self.seq_tag_size,
            drop_prob
        )

        # relation judgement
        self.rel_judgement = MultiNonLinearClassifier(
            config.hidden_size,
            self.rel_num,
            drop_prob
        )
        self.rel_embedding = nn.Embedding(self.rel_num, config.hidden_size)

        self.corres_mode = corres_mode
        if self.corres_mode == 'biaffine':
            self.U = torch.nn.Parameter(
                torch.randn(
                    biaffine_hidden_size,
                    1,
                    biaffine_hidden_size
                )
            )
            self.start_encoder = torch.nn.Sequential(
                torch.nn.Linear(in_features=config.hidden_size,
                                out_features=biaffine_hidden_size),
                torch.nn.ReLU()
            )
            self.end_encoder = torch.nn.Sequential(
                torch.nn.Linear(in_features=config.hidden_size,
                                out_features=biaffine_hidden_size),
                torch.nn.ReLU()
            )
        else:
            # global correspondence
            self.global_corres = MultiNonLinearClassifier(
                config.hidden_size * 2,
                1,
                drop_prob
            )

        self.init_weights()

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        seq_tags=None,
        potential_rels=None,
        rel_threshold=0.1,
        **kwargs

    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            rel_tags: (bs, rel_num)
            potential_rels: (bs,), only in train stage.
            seq_tags: (bs, 2, seq_len)
            corres_tags: (bs, seq_len, seq_len)
            ex_params: experiment parameters
        """

        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        bs, seq_len, h = sequence_output.size()

        # (bs, h)
        h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
        # (bs, rel_num)
        rel_pred = self.rel_judgement(h_k_avg)

        if self.corres_mode == 'biaffine':
            sub_extend = self.start_encoder(sequence_output)
            obj_extend = self.end_encoder(sequence_output)

            corres_pred = torch.einsum('bxi,ioj,byj->bxyo', sub_extend, self.U, obj_extend).squeeze(-1)
        else:
            sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
            obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)
            # batch x seq_len x seq_len x 2*hidden
            corres_pred = torch.cat([sub_extend, obj_extend], 3)
            # (bs, seq_len, seq_len)
            corres_pred = self.global_corres(corres_pred).squeeze(-1)

        # relation predict and data construction in inference stage
        xi, pred_rels = None, None
        if seq_tags is None:
            # (bs, rel_num)
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > rel_threshold,
                                          torch.ones(rel_pred.size(), device=rel_pred.device),
                                          torch.zeros(rel_pred.size(), device=rel_pred.device))

            # if potential relation is null
            for idx, sample in enumerate(rel_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)
                    max_index = torch.argmax(rel_pred[idx])
                    sample[max_index] = 1
                    rel_pred_onehot[idx] = sample

            # 2*(sum(x_i),)
            bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot, as_tuple=True)
            # get x_i
            xi_dict = Counter(bs_idxs.tolist())
            xi = [xi_dict[idx] for idx in range(bs)]

            pos_seq_output = []
            pos_potential_rel = []
            pos_attention_mask = []
            for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                # (seq_len, h)
                pos_seq_output.append(sequence_output[bs_idx])
                pos_attention_mask.append(attention_mask[bs_idx])
                pos_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            sequence_output = torch.stack(pos_seq_output, dim=0)
            # (sum(x_i), seq_len)
            attention_mask = torch.stack(pos_attention_mask, dim=0)
            # (sum(x_i),)
            potential_rels = torch.stack(pos_potential_rel, dim=0)

        # (bs/sum(x_i), h)
        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)

        if self.emb_fusion == 'concat':
            # (bs/sum(x_i), seq_len, 2*h)
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub = self.sequence_tagging_sub(decode_input)
            output_obj = self.sequence_tagging_obj(decode_input)

        elif self.emb_fusion == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub, output_obj = self.sequence_tagging_sum(decode_input)

        if xi is None:
            return output_sub, output_obj, corres_pred, rel_pred
        else:
            return output_sub, output_obj, corres_pred, pred_rels, xi
