import torch
import torch.nn as nn
import torch.optim as optimise
import time

import models
from reinforce import Agent
from transformer import TransformerModel
from embeddings import Dense
from models import Model, RunAgentModel

BATCHES = 2
CONTEXT_LENGTH = 20
DIM = 128
HIDDEN = 1028
HEADS = 8
LOOPS = 2
INPUT_SIZE = 5
HEAD_NEURONS = 256

agent = Model(DIM, DIM * 2, HIDDEN, HEADS, LOOPS)
policy = Agent(DIM * 2, HEAD_NEURONS, 8)
value = Agent(DIM * 2, HEAD_NEURONS, 1)


while True:
    test_context = torch.rand((CONTEXT_LENGTH, BATCHES, DIM * 2))
    test_input = torch.rand((INPUT_SIZE, BATCHES, DIM))

    actions, state_value = RunAgentModel(test_input, test_context, agent, policy, value)

    throttle, steer, pitch, yaw, roll, jump, boost, brake = actions[1]
    print(f'Agent Decision: throttle: {throttle}, steer: {steer}, pitch: {pitch}, yaw: {yaw}, roll: {roll}, jump: {jump}, boost: {boost}, brake: {brake}')
    print(models.TrainingPolicy(actions[1]))
    print(f'Value Network Winning Approximation: {state_value[0]}')
