import torch
import torch.nn as nn
import torch.optim as optimise
import time


class TransformerBlock(nn.Module):
    def __init__(self, dense_neurons, att_heads, dim, model):
        super(TransformerBlock, self).__init__()
        self.mask = True if model == 'decoder' else False
        # Dense parameters
        self.dense1 = nn.Linear(dim, dense_neurons)
        self.dense2 = nn.Linear(dense_neurons, dim)
        self.gelu = nn.GELU()

        # multi head attention params
        self.attention = nn.MultiheadAttention(dim, att_heads, add_zero_attn=self.mask)

        # layer-norm params
        self.norm_att = nn.LayerNorm(dim)
        self.norm_dense = nn.LayerNorm(dim)

    def forward(self, x):
        # run the model

        # run self attention and add + norm
        pre_attention = x
        post_attention, _ = self.attention(x, x, x)
        residual = pre_attention + post_attention
        normalised = self.norm_att(residual)

        # run the MLP then add + norm
        pre_dense = normalised
        y1 = self.gelu(self.dense1(pre_dense))
        post_dense = self.dense2(y1)
        residual = pre_dense + post_dense
        normalised = self.norm_dense(residual)

        return normalised


class TransformerModel(nn.Module):
    def __init__(self, num_iterations, dense_neurons, att_heads, dim, model='encoder'):
        super(TransformerModel, self).__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dense_neurons, att_heads, dim, model) for _ in range(num_iterations)])

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x
