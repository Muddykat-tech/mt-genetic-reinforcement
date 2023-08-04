from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


# Abstract Super Class, used to ensure a population of classes share some base methods
class NeuralNetwork(ABC):
    def __init__(self, agent_parameters):
        self.agent_parameters = agent_parameters

    @abstractmethod
    def forward(self, raw_input: Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_weights_biases(self) -> np.array:
        pass

    @abstractmethod
    def update_weights_biases(self, weights_biases: np.array) -> None:
        pass

    def load(self, file):
        self.update_weights_biases(np.load(file))
