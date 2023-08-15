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
    environment = gym_super_mario_bros.make(environment_name)
    environment = JoypadSpace(environment, SIMPLE_MOVEMENT)
    environment = ConcatObs(env=environment, k=4)
    return environment


def create_mario_environment_random(environment_name='SuperMarioBrosRandomStages-v0'):
    environment = gym_super_mario_bros.make(environment_name)
    environment = JoypadSpace(environment, SIMPLE_MOVEMENT)
    environment = ConcatObs(env=environment, k=4)
    return environment


def test_mario_model(model):
    done = False
    state = env.reset()
    frames = np.zeros(shape=(1, model.agent_parameters['n_frames'], model.agent_parameters['downsample_w'],
                             model.agent_parameters['downsample_h']))
    for episode in range(20000):
        time.sleep(0.005)
        env.render()

        observation = torch.from_numpy(state.copy()).float()
        processed_state = model.preprocess.forward(observation[episode % model.agent_parameters['n_frames'],])
        frames[0, episode % model.agent_parameters['n_frames']] = processed_state

        data = torch.from_numpy(frames).to(model.device)
        action_probability = torch.nn.functional.softmax(model.forward(data).mul(model.agent_parameters['action_conf']),
                                                         dim=1)
        m = torch.distributions.Categorical(action_probability)
        action = m.sample().item()

        for _ in range(model.agent_parameters['n_repeat']):
            obs, reward, done, _ = env.step(action)
            if done:
                break
        if done:
            break


if __name__ == "__main__":
    # Environment Setup
    env = create_mario_environment()

    model_name = '-08-14-2023_21-32_NN=CNNIndividual_POPSIZE=20_GEN=50_PMUTATION_0.05_PCROSSOVER_0.5_BATCH_SIZE=16__I=0_SCORE=327.1333333333233'

    print(env.action_space)

    # Create and load a Model
    model = CNN(AgentParameters.MarioCudaAgent().agent_parameters)
    model.load('../../models/' + model_name + '.npy')

    for _ in range(12):
        env.reset()
        test_mario_model(model)

    env.close()
