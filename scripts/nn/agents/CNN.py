from abc import ABC
from collections import OrderedDict
from typing import Tuple, Any

import numpy as np
import torch

from nn.agents.NeuralNetwork import NeuralNetwork
from torch import nn, Tensor

from nn.preprocess.preproc import Preproc


class CNN(nn.Module, NeuralNetwork):
    def __init__(self, parameters):
        NeuralNetwork.__init__(self, parameters)
        super(CNN, self, ).__init__()
        self.fitness = 0.0
        gpu = self.agent_parameters["gpu"]
        self.device = torch.device("cuda" if gpu >= 1 else "cpu")
        self.preprocess = Preproc(self.agent_parameters)

        input_size = self.agent_parameters['input_size']
        hidden_size = self.agent_parameters['hidden_size']
        output_size = self.agent_parameters['output_size']

        # Based on Deep Q Learning's CNN Settings (By default hidden_size is 32
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=8, stride=4).to(
            self.device)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=3, stride=1).to(self.device)
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512).to(self.device)
        self.fc2 = nn.Linear(in_features=512, out_features=output_size).to(self.device)

        self.relu = nn.ReLU().to(self.device)

    def get_fitness(self):
        return self.fitness

    def forward(self, raw_input: Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(raw_input.float()))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_weights_biases(self) -> np.array:
        parameters = self.state_dict().values()
        parameters = [p.flatten() for p in parameters]
        parameters = torch.cat(parameters, 0)
        return parameters.cpu().detach().numpy()

    def update_weights_biases(self, weights_biases: np.array) -> None:
        weights_biases = torch.from_numpy(weights_biases)
        shapes = [x.shape for x in self.state_dict().values()]
        shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

        partial_split = weights_biases.split(shapes_prod)
        model_weights_biases = []
        for i in range(len(shapes)):
            model_weights_biases.append(partial_split[i].view(shapes[i]))
        state_dict = OrderedDict(zip(self.state_dict().keys(), model_weights_biases))
        self.load_state_dict(state_dict)
