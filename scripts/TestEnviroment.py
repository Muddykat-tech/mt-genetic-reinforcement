import random
from collections import deque

import gym
import torch
from gym import spaces
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
from gaopenai.nn import rnn
from gaopenai.nn.rnn import RNN


def get_weighted_action(action_probabilities):
    action_probabilities = action_probabilities.flatten()
    weighted_indices = []
    for i, val in enumerate(action_probabilities):
        weight = val ** 2  # Adjust the weighting here (e.g., val ** 2, val ** 3)
        weighted_indices.extend([i] * int(weight * 100))  # Scale the weights as desired

    # Randomly select an index from the weighted indices list
    random_index = random.choice(weighted_indices)
    return random_index


class ConcatObs(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = \
            spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)

        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.array(self.frames)


def test_rnn(input_size=None, is_reduced=False):
    state = env.reset()
    rnn.init_hidden()
    start = True
    frames = np.zeros(shape=(32, 4, rnn.downsample_w, rnn.downsample_h))
    for episode in range(20000):
        env.render()
        observation = torch.from_numpy(state.copy()).float()

        # This section needs to be reworked, unneeded the obs has the frames needed through the wrapper.
        if start:
            for _ in range(4):
                processed_state = rnn.preprocess.forward(observation)
                frames[_, 0] = processed_state
        else:
            processed_state = rnn.preprocess.forward(observation)
            frames[episode % 4, 0] = processed_state
        #
        data = torch.from_numpy(frames).to(rnn.device)
        action = rnn.forward(data)
        actions = np.array(action.cpu().detach().numpy())

        action = get_weighted_action(actions) % 7

        for _ in range(4):
            obs, reward, done, _ = env.step(action)
            if done:
                break
        if done:
            break


env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = ConcatObs(env=env, k=4)

# Downsample, Greyscale and use CNN
print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)

INPUT_SIZE = 4
HIDDEN_SIZE = 32
OUTPUT_SIZE = 7

rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
rnn.load('../models/' + '07-24-2023_16-06_NN=RNNIndividual_POPSIZE=20_GEN=80_PMUTATION_0.6_PCROSSOVER_0.5' + '.npy')

for _ in range(3):
    env.reset()
    test_rnn(input_size=INPUT_SIZE, is_reduced=True)

env.close()
