import time

import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from environment.util.EnviromentUtil import ConcatObs, get_weighted_action
from nn.agents.CNN import CNN
from nn.setup import AgentParameters


# Functions
def create_mario_environment(environment_name='SuperMarioBros-v0'):
    env = gym_super_mario_bros.make(environment_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ConcatObs(env=env, k=4)
    return env


def test_mario_model(model):
    done = False
    state = env.reset()
    frames = np.zeros(shape=(32, 4, model.agent_parameters['downsample_w'], model.agent_parameters['downsample_h']))
    for episode in range(20000):
        time.sleep(0.01)
        env.render()
        observation = torch.from_numpy(state.copy()).float()
        processed_state = model.preprocess.forward(observation[episode % 4, ])
        frames[episode % 4, 0] = processed_state

        data = torch.from_numpy(frames).to(model.device)
        action = model.forward(data)
        actions = np.array(action.cpu().detach().numpy())

        action = get_weighted_action(actions) % 7

        for _ in range(4):
            obs, reward, done, _ = env.step(action)
            if done:
                break
        if done:
            break


if __name__ == "__main__":
    # Environment Setup
    env = create_mario_environment()

    model_name = '07-24-2023_16-06_NN=RNNIndividual_POPSIZE=20_GEN=80_PMUTATION_0.6_PCROSSOVER_0.5'

    # Create and load a Model
    model = CNN(AgentParameters.MarioCudaAgent().agent_parameters)
    model.load('../../models/' + model_name + '.npy')

    for _ in range(3):
        env.reset()
        test_mario_model(model)

    env.close()
