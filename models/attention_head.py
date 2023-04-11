from math import sqrt

import torch
import torch.nn as nn
class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att
#  import torch
# from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm


# class TransformerEncoderLayer(Module):
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).
#     Examples::
#     """

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = Linear(d_model, dim_feedforward)
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model)

#         self.norm1 = LayerNorm(d_model)
#         self.norm2 = LayerNorm(d_model)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#         import torch.nn.functional as F
#         self.activation = F.relu

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             import torch.nn.functional as F
#             state['activation'] = F.relu
#         super(TransformerEncoderLayer, self).__setstate__(state)

#     def forward(self, src):
#         r"""Pass the input through the encoder layer.
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#         Shape:
#             see the docs in Transformer class.
#         """
#         src2, att_weights = self.self_attn(src, src, src, attn_mask=None,
#                                            key_padding_mask=None)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src, att_weights

# def obtain_sample_relation_models(add_bt, output_dim, debug=0):
#     """
#     This is for ablation study
#     debug is for attention weights.
#     """
#     if debug == 2:
#         encoder_layers = TransformerEncoderLayer(output_dim, 4, output_dim, 0)
#         return encoder_layers
#     if add_bt == 2:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.5)
#         encoder_layers = torch.nn.TransformerEncoder(encoder_layers, 2)
#     elif add_bt == 3:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.1)
#     elif add_bt == 4:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 8, output_dim, 0.5)
#     elif add_bt == 6:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 1.)
#     elif add_bt == 7:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.5)
#         encoder_layers = torch.nn.TransformerEncoder(encoder_layers, 4)
#     elif add_bt == 8:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 1, output_dim, 0.5)
#     elif add_bt == 12:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.5)
#         encoder_layers = torch.nn.TransformerEncoder(encoder_layers, 8)

#     elif add_bt == 14:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.5)
#         encoder_layers = torch.nn.TransformerEncoder(encoder_layers, 16)
#     elif add_bt == 15:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.5)
#         encoder_layers = torch.nn.TransformerEncoder(encoder_layers, 32)
#     elif add_bt == 13:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.5)
#         encoder_layers = torch.nn.TransformerEncoder(encoder_layers, 6)
#     elif add_bt == 18:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.)
#     elif add_bt == 19:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.7)
#     else:
#         encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, 4, output_dim, 0.5)
#     return  encoder_layers