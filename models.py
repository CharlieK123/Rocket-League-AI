
import torch
import torch.nn as nn
import torch.optim as optimise
import time
import numpy as np

from transformer import TransformerModel
from embeddings import Dense
from reinforce import Agent

import random

# Constant parameters
GAME_STATE_VALUES = 8
BALL_VALUES = 3
GOAL_VALUES = 2
BOOST_VALUES = 4
PLAYER_VALUES = 21
ACTION_STATES = 8
states = [3, 2, 4, 21, 8]


class Model(nn.Module):
    def __init__(self, e_dim, d_dim, neurons, heads, transformer_itr):
        # input embeddings
        super(Model, self).__init__()
        self.ball_embedding = Dense(BALL_VALUES, 20, e_dim)
        self.goal_embedding = Dense(GOAL_VALUES, 20, e_dim)
        self.boost_embedding = Dense(BOOST_VALUES, 20, e_dim)
        self.player_embedding = Dense(PLAYER_VALUES, 20, e_dim)
        self.embeddings = [self.ball_embedding, self.goal_embedding, self.boost_embedding, self.player_embedding]

        # state encoding
        self.game_state_encoder = Agent(GAME_STATE_VALUES, 128, e_dim)
        self.entity_encoder = TransformerModel(transformer_itr, neurons, heads, e_dim, model='encoder')

        # state decoder
        # NOTE: DIM will likely vary in the decoder, testing should finalise this value
        self.decoder_embedding = Agent(d_dim, 128, d_dim)  # not typical in transformers possible weakness
        self.game_state_decoder = TransformerModel(transformer_itr, neurons, heads, d_dim, model='decoder')

    def forward(self, x, context):
        # Run the entire model x = (Batch, num_inputs, Dim)

        # start by passing the game attributes through a FNN
        game_features = x[-1][:, :states[-1]]
        game_encodings = self.game_state_encoder(game_features, model='normal')

        # embed the data for the entity encoder
        entity_embeddings = torch.zeros_like(x)
        for i, vector in enumerate(x[:-1]):
            x1 = vector[:, :states[i]]
            y1 = self.embeddings[i](x1)
            entity_embeddings[i] = y1

        # encode the entities with a encoder transformer
        entity_encodings = self.entity_encoder(entity_embeddings)

        # average the values at dim 0 to transform to a batch size matrix
        entity_encodings = entity_encodings.mean(dim=0)

        # combine the game state encoding and the entity encoding into one vector
        state_embedding = torch.cat((entity_encodings, game_encodings), dim=1).unsqueeze(0)
        decoder_state_context = torch.cat((context, state_embedding), dim=0)

        # run the decision decoder
        decoded_state = self.game_state_decoder(decoder_state_context)
        decoded_current_state = decoded_state[-1]

        return decoded_current_state


def RunAgentModel(input, context, model, policy, value):
    decoded_output = model(input, context)

    policy_out = policy(decoded_output, model='policy')
    value_out = value(decoded_output, model='value')

    return policy_out, value_out


def TrainingPolicy(output):
    output = output.detach()
    throttle, steer, pitch, yaw, roll = np.array(output[:5])
    jump, boost, brake = np.array(output[5:])
    print(jump)

    do_jump = 1 if random.random() <= jump else 0
    do_boost = 1 if random.random() <= boost else 0
    do_brake = 1 if random.random() <= brake else 0

    rand_throttle = np.random.normal(throttle, 0.1)
    rand_steer = np.random.normal(steer, 0.1)
    rand_pitch = np.random.normal(pitch, 0.1)
    rand_yaw = np.random.normal(yaw, 0.1)
    rand_roll = np.random.normal(roll, 0.1)

    print([rand_throttle, rand_steer, rand_pitch, rand_yaw, rand_roll, do_jump, do_boost, do_brake])
    return rand_throttle, rand_steer, rand_pitch, rand_yaw, rand_roll, do_jump, do_boost, do_brake
