"""
Additional layers.
"""
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils


class LSTMLayer(nn.Module):
    """ A wrapper for LSTM with sequence packing. """

    def __init__(self, emb_dim, hidden_dim, num_layers, dropout, use_cuda):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(
            emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.use_cuda = use_cuda

    def forward(self, x, x_mask, init_state):
        """
        x: batch_size * feature_size * seq_len
        x_mask : batch_size * seq_len
        """
        x_lens = x_mask.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lens = list(x_lens[idx_sort])

        # sort by seq lens
        x = x.index_select(0, idx_sort)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        rnn_output, (ht, ct) = self.rnn(rnn_input, init_state)
        rnn_output = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)[0]

        # unsort
        rnn_output = rnn_output.index_select(0, idx_unsort)
        ht = ht.index_select(0, idx_unsort)
        ct = ct.index_select(0, idx_unsort)
        return rnn_output, (ht, ct)


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    rat_size = rationale embedding size
    """

    def __init__(self, input_size, query_size, feature_size, attn_size, rat_size=None):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.rat_size = rat_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        
        if rat_size is not None:
            self.rlinear = nn.Linear(
                rat_size, attn_size, bias=False
            )  # for rationale
        else:
            self.rlinear = None

        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        if self.rlinear is not None:
            self.rlinear.weight.data.normal_(std=0.01)
        self.tlinear.weight.data.zero_()  # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask, q, f, rationale=None):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()
        # print(x.size(), rationale.size(), f.size())

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size
        )
        q_proj = (
            self.vlinear(q.view(-1, self.query_size))
            .contiguous()
            .view(batch_size, self.attn_size)
            .unsqueeze(1)
            .expand(batch_size, seq_len, self.attn_size)
        )
        if self.wlinear is not None:
            f_proj = (
                self.wlinear(f.view(-1, self.feature_size))
                .contiguous()
                .view(batch_size, seq_len, self.attn_size)
            )
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]

        # if self.rlinear is not None:
        #     r_proj = (
        #         self.rlinear(rat_features.view(-1, self.rat_size))
        #         .contiguous()
        #         .view(batch_size, seq_len, self.attn_size)
        #     )
        #     projs.append(r_proj)

        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len
        )

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float("inf"))
        
        if rationale is not None:
            # clear non-rationale tokens for attention
            scores.data.masked_fill_((1-rationale).bool().data, -float("inf"))

        weights = F.softmax(scores, dim=1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs, weights
