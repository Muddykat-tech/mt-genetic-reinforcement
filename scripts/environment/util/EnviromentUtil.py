import random
from collections import deque

import gym
import numpy as np
import torch
from gym import spaces

from nn.preprocess import preproc


# replace with softmax
def get_weighted_action(action_probabilities):
    action_probabilities = action_probabilities.flatten()
    weighted_indices = []
    random_index = 0
    for i, val in enumerate(action_probabilities):
        weight = val ** 2  # Adjust the weighting here (e.g., val ** 2, val ** 3)
        weighted_indices.extend([i] * int(weight * 100))  # Scale the weights as desired

    if len(weighted_indices) != 0:
        # Randomly select an index from the weighted indices list
        random_index = random.choice(weighted_indices)
    return random_index


class ConcatObs(gym.Wrapper):
    def __init__(self, env, k, frame_skip):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = \
            spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)
        self.frame_skip = frame_skip

    def reset(self):
        ob = self.env.reset()
        ob, _, _, self.info = self.env.step(0)
        ob = torch.from_numpy(ob.copy()).float()
        ob = preproc.forward(True, 84, 84, ob)

        for _ in range(self.k):
            self.frames.append(ob)

        return self._get_ob(), self.info

    def step(self, action):

        reward = 0
        for _ in range(self.frame_skip):
            ob, tmp_reward, done, self.info = self.env.step(action)
            reward += tmp_reward

            if done:
                break

        ob = torch.from_numpy(ob.copy()).float()
        ob = preproc.forward(True, 84, 84, ob)
        self.frames.append(ob)

        return self._get_ob(), reward, done, self.info

    def _get_ob(self):
        return torch.stack(list(self.frames)).unsqueeze(0)
