import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from transformers import BertPreTrainedModel


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        ent_sub = self.dropout(self.mlp1(x))
        ent_obj = self.dropout(self.mlp2(y))
        o1 = self.biaffine(ent_sub, ent_obj)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class W2NERBert(BertPreTrainedModel):
    """
    W2NER Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        use_bert_last_4_layers (:obj:`bool`, optional, defaults to True)：Encoder Layer bert层数
        dist_emb_size (:obj:`int`, defaults to 20)：
            Convolution Layer - Distance Embedding 用于表示距离向量的大小
        type_emb_size (:obj:`int`, defaults to 20)：
            Convolution Layer - Rigion Embedding 用于表示上下矩阵向量的大小
        lstm_hid_size (:obj:`int`, defaults to 512)：
            Convolution Layer - Word Embedding 用于表示词向量的大小, defaults to 768(large)
        conv_hid_size (:obj:`int`, defaults to 96)：Convolution Layer 卷积的大小
        biaffine_size (:obj:`int`, defaults to 512)：
            Co-Predictor Layer biaffine的大小, defaults to 768(large)
        ffnn_hid_size (:obj:`int`, defaults to 288)：
            Co-Predictor Layer MLP的大小, defaults to 128(large)
        dilation (:obj:`list`, defaults to [1, 2, 3]): Convolution Layer - Dialation Rate 类似textcnn
        conv_dropout (:obj:`float`, defaults to 0.5): Convolution Layer dropout
        emb_dropout (:obj:`float`, defaults to 0.5): Encoder Layer dropout
        out_dropout (:obj:`float`, defaults to 0.33): Co-Predictor Layer dropout
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://github.com/ljynlp/w2ner
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        use_bert_last_4_layers=True,

        dist_emb_size=20,
        type_emb_size=20,
        lstm_hid_size=512,
        conv_hid_size=96,

        biaffine_size=512,
        ffnn_hid_size=288,
        dilation=[1, 2, 3],

        conv_dropout=0.5,
        emb_dropout=0.5,
        out_dropout=0.33,
        ** kwargs
    ):
        super(W2NERBert, self).__init__(config)
        self.num_labels = config.num_labels

        self.use_bert_last_4_layers = use_bert_last_4_layers

        self.lstm_hid_size = lstm_hid_size
        self.conv_hid_size = conv_hid_size

        lstm_input_size = 0

        self.bert = BertModel(config)
        lstm_input_size += config.hidden_size

        self.dis_embs = nn.Embedding(20, dist_emb_size)
        self.reg_embs = nn.Embedding(3, type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        conv_input_size = lstm_hid_size + dist_emb_size + type_emb_size

        self.convLayer = ConvolutionLayer(conv_input_size, conv_hid_size, dilation, conv_dropout)
        self.dropout = nn.Dropout(emb_dropout)
        self.predictor = CoPredictor(self.num_labels, lstm_hid_size, biaffine_size,
                                     conv_hid_size * len(dilation), ffnn_hid_size,
                                     out_dropout)

        self.cln = LayerNorm(lstm_hid_size, lstm_hid_size, conditional=True)

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            grid_mask2d,
            dist_inputs,
            pieces2word,
            input_lengths,
            **kwargs
    ):
        bert_embs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        # 源码每个batch的长度等于 input_lengths.max()
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=length)

        cln = self.cln(word_reps.unsqueeze(2), word_reps)

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)

        return outputs
