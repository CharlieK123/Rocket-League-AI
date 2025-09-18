import torch
import numpy
import torch.nn as nn
import torch.optim as optimise
from embeddings import Dense
import math
from reinforce import Agent
import torch.nn.functional as F
import time

GAME_STATE_VALUES = 8
BALL_VALUES = 3
GOAL_VALUES = 2
BOOST_VALUES = 4
PLAYER_VALUES = 21
ACTION_STATES = 8  # number of output neurons (different actions to be taken)

dim = 64  # global encoder vector dimensionality
emb_hidden = 48  # obs embedding network hidden neuron count
attention_heads = 4  # amount of heads used in MHA
dense_hidden = 128  # encoder dense network hidden neuron count
blocks = 1  # amount of encoder iterations done
max_len = 6  # maximum number of context tokens used in one given input
out_hidden = 128  # actor/critic network hidden neuron count

test_ball = torch.rand((1, BALL_VALUES))
test_goal = torch.rand((1, GOAL_VALUES))
test_boost = torch.rand((1, BOOST_VALUES))
test_player = torch.rand((1, PLAYER_VALUES))
test_opp = torch.rand((1, PLAYER_VALUES))
test_game_state = torch.rand((1, GAME_STATE_VALUES))


class TokenAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(max_len))

    def forward(self, x):
        # x: (seq, batch, dim)
        weights = F.softmax(self.attention_weights, dim=0)  # (seq,)
        weights = weights.unsqueeze(1).unsqueeze(2)  # (seq, 1, 1)
        weighted = (x * weights).sum(dim=0)  # (batch, dim)
        return weighted


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # apply pos encoding
        x = x + self.pe[:x.size(0)]
        return x


class Encoder(nn.Module):
    def __init__(self):
        # input embeddings
        super(Encoder, self).__init__()
        # four attention modules
        self.attention = nn.MultiheadAttention(dim, attention_heads)

        # four feedforward modules
        self.FFN = Dense(dim, dense_hidden, dim)

        # four normalisation modules
        self.layernorm_att = nn.LayerNorm(dim)
        self.layernorm_dense = nn.LayerNorm(dim)

    def forward(self, x):
        # attention layer
        pre_att = x
        post_att, _ = self.attention(x, x, x)
        residual = pre_att + post_att

        # first norm layer
        x = self.layernorm_att(residual)

        # dense layer
        pre_dense = x
        post_dense = self.FFN(x)
        residual = pre_dense + post_dense

        # second norm layer
        x = self.layernorm_dense(residual)

        return x


class Model(nn.Module):
    def __init__(self, role):
        super(Model, self).__init__()

        # encode input obs to input vector
        # each type of input gets their own embedding network
        self.ball_embedding = Dense(BALL_VALUES, emb_hidden, dim)
        self.goal_embedding = Dense(GOAL_VALUES, emb_hidden, dim)
        self.boost_embedding = Dense(BOOST_VALUES, emb_hidden, dim)
        self.player_embedding = Dense(PLAYER_VALUES, emb_hidden, dim)
        self.opponent_embedding = Dense(PLAYER_VALUES, emb_hidden, dim)
        self.game_feats_embedding = Dense(GAME_STATE_VALUES, emb_hidden, dim)

        # positional encoding params
        self.pos_encoding = PositionalEncoding()

        # encoder model (n iterations)
        self.encoder_blocks = nn.ModuleList([Encoder() for _ in range(blocks)])

        # aggregate encoder out
        self.aggregator = TokenAggregator()

        if role == 'actor':
            self.output = Agent(dim, out_hidden, ACTION_STATES, role='actor')
        else:  # assume critic role
            self.output = Agent(dim, out_hidden, 1, role='critic')

    def forward(self, ball_vec, goal_vec, boost_vec, player_vec, opponent_vec, feats_vec):

        # encode all the different obs inputs to the same dim
        encoded_ball_vec = self.ball_embedding(ball_vec)
        encoded_goal_vec = self.goal_embedding(goal_vec)
        encoded_boost_vec = self.boost_embedding(boost_vec)
        encoded_player_vec = self.player_embedding(player_vec)
        encoded_opponent_vec = self.opponent_embedding(opponent_vec)
        encoded_feats_vec = self.game_feats_embedding(feats_vec)

        all_inputs = [encoded_ball_vec, encoded_goal_vec, encoded_boost_vec, encoded_player_vec, encoded_opponent_vec,
                      encoded_feats_vec]
        x = torch.stack(all_inputs, dim=0)  # shape (seq_len, batch, dim)

        # apply positional encoding
        x = self.pos_encoding.forward(x)

        # run the encoder blocks
        for layer in self.encoder_blocks:
            x = layer(x)

        # aggregate encoder output to shape (batch, dim)
        x = self.aggregator.forward(x)

        # perform either policy or value network evaluation
        x = self.output.forward(x)

        return x


agent = Model(role='actor')
prev_time = time.time()
sps = 0

while True:
    actions = agent.forward(test_ball, test_goal, test_boost, test_player, test_opp, test_game_state)
    sps += 1
    if time.time() - prev_time >= 1:
        print(f'---model run {sps}itr/s---')
        prev_time = time.time()
        sps = 0
