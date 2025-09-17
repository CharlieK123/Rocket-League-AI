import rlgym
import torch
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
import torch.nn as nn
import torch.optim as optimise
import time
import models
from reinforce import Agent
from transformer import TransformerModel
from embeddings import Dense
from models import Model, RunAgentModel
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.obs_builders import ObsBuilder
from reinforce import TrainModel

BATCHES = 2
CONTEXT_LENGTH = 20
DIM = 128
HIDDEN = 1028
HEADS = 8
LOOPS = 2
INPUT_SIZE = 5
HEAD_NEURONS = 256
EPOCHS = 10

d = torch.tensor([0.99])
l = 0.95
e = 0.2

agent = Model(DIM, DIM * 2, HIDDEN, HEADS, LOOPS)
policy = Agent(35, HEAD_NEURONS, 8)
value = Agent(35, HEAD_NEURONS, 1)

test_context = torch.rand((CONTEXT_LENGTH, BATCHES, DIM * 2))
test_input = torch.rand((INPUT_SIZE, BATCHES, DIM))

default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 20

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

condition1 = TimeoutCondition(max_steps)


class CustomObsBuilder(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        # Ball info
        ball = state.ball
        ball_vec = np.concatenate([
            ball.position / 4000,
            ball.linear_velocity / 2300,
            ball.angular_velocity / 5
        ])

        # Self info
        self_car = player.car_data
        self_vec = np.concatenate([
            self_car.position / 4000,
            self_car.linear_velocity / 2300,
            [player.boost_amount / 100],
            self_car.forward(),
            self_car.up()
        ])

        # Opponent info (if exists, else zeros)
        opponents = [p for p in state.players if p.team_num != player.team_num]
        if len(opponents) > 0:
            opp_player = opponents[0]
            opp_car = opp_player.car_data
            opp_vec = np.concatenate([
                opp_car.position / 4000,
                opp_car.linear_velocity / 2300,
                [opp_player.boost_amount / 100],
                opp_car.forward(),
                opp_car.up()
            ])
        else:
            opp_vec = np.zeros_like(self_vec)  # placeholder of same size

        # Combine
        obs = np.concatenate([ball_vec, self_vec, opp_vec])
        obs = torch.tensor([obs], dtype=torch.float32)
        return obs


env = rlgym.make(terminal_conditions=[condition1], team_size=1, spawn_opponents=True, obs_builder=CustomObsBuilder(), self_play=True)
test_obs = env.reset()


while True:
    print('running')
    steps = 0
    saved_rewards, saved_values = [], []
    old_actions = []
    obs = env.reset()
    done = False

    while not done:
        # Here we sample a random action. If you have an agent, you would get an action from it here.
        print(f'step: {steps}')
        agent_actions = []
        for i in range(len(obs)):
            game_state = obs[i]
            print(game_state.shape)
            action_vector, state_eval = policy(game_state, model='policy'), value(game_state, model='value')
            agent_actions.append((action_vector, state_eval))
            action_vector = action_vector.detach()
            action = torch.tensor(action_vector)

            print(f'Agent: {i}, input: {game_state.shape}, policy: {action_vector}, value: {state_eval}')

        next_obs, reward, done, gameinfo = env.step(action)
        obs = next_obs
        steps += 1


