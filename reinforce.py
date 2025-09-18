import torch
import torch.nn as nn
import torch.optim as optimise
import time

ACTION_SPACE = 8
NEURONS = 128
DIM = 512



class Agent(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_size, role):
        # define the parameters in the model
        # defines a model with two hidden layers of n neurons
        super(Agent, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_neurons)
        self.hidden_layer2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.hidden_layer3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.hidden_layer4 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, output_size, bias=False)

        # define the activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.role = False
        if role == 'actor': self.role = True

    def forward(self, x):
        # perform the operations within the model
        y1 = self.relu(self.hidden_layer1(x))
        y2 = self.relu(self.hidden_layer2(y1))
        y3 = self.relu(self.hidden_layer3(y2))
        y4 = self.relu(self.hidden_layer4(y3))
        out = self.output_layer(y4)

        if self.role:
            out = self.tanh(out)

        return out


def Reward(rewards, steps, discount):
    return torch.sum(rewards * (discount ** torch.arange(steps)))


def TemporalDifference(rewards, values, discount):
    return rewards + (discount * values[1:]) - values[:-1]


def GAE(rewards, values, discount_factor, trace_decay, normalize=True):
    advantages = []
    advantage = 0
    next_value = 0

    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * discount_factor - v
        advantage = td_error + advantage * discount_factor * trace_decay
        next_value = v
        advantages.insert(0, advantage)

    advantages = torch.tensor(advantages)

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def GetRatio(new_vector, old_vector):
    new_vec_diff = torch.norm(new_vector - old_vector, dim=-1)
    old_vec = torch.norm(old_vector, dim=-1)

    return new_vec_diff / (old_vec + 1e-8)


def ClippedSurrogateLoss(advantage, old_prob, new_prob, elip):
    ratio = GetRatio(new_prob, old_prob)
    return torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - elip, 1 + elip) * advantage)


def PolicyLoss(all_old_prob, all_new_prob, all_values, all_rewards, steps, discount=0.99, lmbda=0.95, elip=0.2):
    losses = torch.tensor([0.0], requires_grad=True)
    advantage = GAE(all_rewards, all_values, discount, lmbda)
    print(advantage)

    for i in range(steps):
        new_prob, old_prob = all_new_prob[i], all_old_prob[i]
        loss = ClippedSurrogateLoss(advantage[i], old_prob, new_prob, elip)
        losses = losses + loss

    return losses / steps


def ValueLoss(all_values, all_rewards, steps, discount=0.99):
    losses = torch.tensor([0.0], requires_grad=True)

    for i in range(steps):
        future_rewards = all_rewards[i:]
        print(future_rewards)
        total_reward = Reward(future_rewards, steps - i, discount)
        loss = (all_values[i] - total_reward) ** 2
        losses = losses + loss

    return losses / steps


def TrainModel(model, policy_net, value_net, rewards, values, old_actions, predicted_actions, steps, d, l, e, lr):
    optimiser = optimise.Adam(list(model.parameters()) + list(policy_net.parameters()) + list(value_net.parameters()),
                              lr=lr)
    optimiser.zero_grad()

    policy_loss = PolicyLoss(old_actions, predicted_actions, values, rewards, steps, d, l, e)
    value_loss = ValueLoss(values, rewards, steps, d)

    combined_loss = policy_loss + value_loss

    combined_loss.backward()
    optimiser.step()
