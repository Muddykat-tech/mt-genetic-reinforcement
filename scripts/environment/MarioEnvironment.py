import time

import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from environment.util.EnviromentUtil import ConcatObs
from ga.components.Individuals import CNNIndividual
from nn.agents.CNN import CNN
from nn.setup import AgentParameters


# Functions
def create_mario_environment(environment_name='SuperMarioBros-1-1-v0'):
    environment = gym_super_mario_bros.make(environment_name)
    environment = JoypadSpace(environment, COMPLEX_MOVEMENT)
    environment = ConcatObs(env=environment, k=4, frame_skip=8)
    return environment


def create_mario_environment_random(environment_name='SuperMarioBrosRandomStages-v0'):
    environment = gym_super_mario_bros.make(environment_name, stages=['1-1', '2-1', '3-1', '4-1'])
    environment = JoypadSpace(environment, COMPLEX_MOVEMENT)
    environment = ConcatObs(env=environment, k=4, frame_skip=8)
    return environment