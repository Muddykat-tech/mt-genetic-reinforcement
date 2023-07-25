from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch

from gaopenai.preprocess.preproc import Preproc
from scripts.gaopenai.nn.base_nn import NeuralNetwork


class RNN(nn.Module, NeuralNetwork):
    def __init__(self, input_size, hidden_size1, output_size):
        super(RNN, self).__init__()
        self.fitness = 0.0
        self.hidden_size1 = hidden_size1
        agent_params = {}
        agent_params["gpu"] = 0
        agent_params["use_rgb_for_raw_state"] = True
        agent_params["downsample_w"] = 84
        agent_params["downsample_h"] = 84
        gpu = agent_params["gpu"]

        self.device = torch.device("cuda" if gpu >= 0 else "cpu")
        self.agent_params = agent_params
        self.downsample_w = agent_params["downsample_w"]
        self.downsample_h = agent_params["downsample_h"]
        self.preprocess = Preproc(self.agent_params, self.downsample_w, self.downsample_h)

        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=8, stride=4).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=2, stride=2).to(self.device)
        self.fc1 = nn.Linear(in_features=800, out_features=32).to(self.device)
        self.fc2 = nn.Linear(in_features=32, out_features=output_size).to(self.device)

        self.relu = nn.ReLU().to(self.device)

    def forward(self, raw_input) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.relu(self.conv1(raw_input.float()))
        input = self.relu(self.conv2(input))
        input = input.view(input.size(0), -1)
        fc1 = self.fc1(input)
        input = self.relu(fc1)
        output = self.fc2(input)
        return output

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(self.hidden_size1)

    def get_weights_biases(self) -> np.array:
        parameters = self.state_dict().values()
        parameters = [p.flatten() for p in parameters]
        parameters = torch.cat(parameters, 0)
        return parameters.detach().numpy()

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
