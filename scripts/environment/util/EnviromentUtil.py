import random
from collections import deque

import gym
import numpy as np
from gym import spaces


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


