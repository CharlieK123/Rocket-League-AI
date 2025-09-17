import torch
import torch.nn as nn
import torch.optim as optimise
import time


class Dense(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_size):
        # define the parameters in the model
        # defines a model with two hidden layers of n neurons
        super(Dense, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_neurons)
        self.hidden_layer2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, output_size, bias=False)

        # define the activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, model='embed'):
        # perform the operations within the model
        y1 = self.relu(self.hidden_layer1(x))
        y2 = self.relu(self.hidden_layer2(y1))
        out = self.output_layer(y2)

        if model != 'embed':
            out = self.softmax(out)

        return out
