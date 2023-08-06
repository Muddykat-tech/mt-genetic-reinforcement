from abc import ABC

from torch import nn

from nn.agents.NeuralNetwork import NeuralNetwork


class DQN(nn.Module, NeuralNetwork):

    def __init__(self, parameters):
        NeuralNetwork.__init__(self, parameters)
        super(DQN, self).__init__()
        input_size = parameters['input_size']
        output_size = parameters['output_size']
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        return self.layer3(x)