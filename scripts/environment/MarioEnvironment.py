import time

import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from environment.util.EnviromentUtil import ConcatObs
from nn.agents.CNN import CNN
from nn.setup import AgentParameters


# Functions
def create_mario_environment(environment_name='SuperMarioBros-v0'):
    environment = gym_super_mario_bros.make(environment_name)
    environment = JoypadSpace(environment, SIMPLE_MOVEMENT)
    environment = ConcatObs(env=environment, k=4, frame_skip=4)
    return environment


def create_mario_environment_random(environment_name='SuperMarioBrosRandomStages-v0'):
    environment = gym_super_mario_bros.make(environment_name)
    environment = JoypadSpace(environment, SIMPLE_MOVEMENT)
    environment = ConcatObs(env=environment, k=4, frame_skip=4)
    return environment


def test_mario_model(agent):
    done = False
    state = env.reset()
    for episode in range(20000):
        time.sleep(0.005)
        env.render()
        state = state.to(agent.device)
        action_probability = torch.nn.functional.softmax(agent.forward(state).mul(agent.agent_parameters['action_conf']),
                                                         dim=1)
        m = torch.distributions.Categorical(action_probability)
        action = m.sample().item()

        for _ in range(agent.agent_parameters['n_repeat']):
            state, reward, done, _ = env.step(action)
            if done:
                break
        if done:
            break


if __name__ == "__main__":
    # Environment Setup
    env = create_mario_environment()

    model_name = '-09-03-2023_05-58_NN=ReinforcementCNNIndividual_POPSIZE=8_GEN=10_PMUTATION_0.05_PCROSSOVER_0.8_BATCH_SIZE=32__I=0_SCORE=240.13333333333384'

    print(env.action_space)

    # Create and load a Model
    model = CNN(AgentParameters.MarioCudaAgent().agent_parameters)
    model.load('../../models/' + model_name + '.npy')

    for _ in range(12):
        env.reset()
        test_mario_model(model)

    env.close()
