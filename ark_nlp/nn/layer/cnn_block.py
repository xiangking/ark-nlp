import torch
import torch.nn as nn
import torch.nn.functional as F


def match_score(s1, s2, mask1, mask2):
    '''
    s1, s2:  batch_size * seq_len  * dim
    '''
    batch, seq_len, dim = s1.shape
    s1 = s1 * mask1.eq(0).unsqueeze(2).float()
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()
    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len, 1)
    s2 = s2.unsqueeze(1).repeat(1, seq_len, 1, 1)
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)


def attention_avg_pooling(sent1, sent2, mask1, mask2):
    # A: batch_size * seq_len * seq_len
    A = match_score(sent1, sent2, mask1, mask2)
    weight1 = torch.sum(A, -1)
    weight2 = torch.sum(A.transpose(1, 2), -1)
    s1 = sent1 * weight1.unsqueeze(2)
    s2 = sent2 * weight2.unsqueeze(2)
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2


class Wide_Conv(nn.Module):
    def __init__(
        self,
        seq_len,
        embeds_size
    ):
        super(Wide_Conv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        self.W = nn.Parameter(torch.randn((seq_len, embeds_size)))
        nn.init.xavier_normal_(self.W)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=[1, 1], stride=1)
        self.tanh = nn.Tanh()

    def forward(
        self,
        sent1,
        sent2,
        mask1,
        mask2
    ):

        A = match_score(sent1, sent2, mask1, mask2)

        attn_feature_map1 = A.matmul(self.W)
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)

        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1)
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2
