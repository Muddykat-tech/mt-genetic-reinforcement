import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch

from environment.util import LoadingLog
from environment.util.EnviromentUtil import get_weighted_action
from nn.agents.CNN import CNN
from nn.agents.NeuralNetwork import NeuralNetwork


# Abstract Individual Class that has base functions the GA can call upon
class Individual(ABC):
    def __init__(self, parameters):
        self.nn = self.get_model(parameters)
        self.fitness = 0.0
        self.weights_biases: np.array = None

    def calculate_fitness(self, env, logger, render) -> None:
        self.fitness, self.weights_biases = self.run_single(env, logger, render)

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)

    @abstractmethod
    def get_model(self, parameters) -> NeuralNetwork:
        pass

    @abstractmethod
    def run_single(self, env, logger, render=False) -> Tuple[float, np.array]:
        pass


# Convolutional Neural Network Individual
class CNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)

    def get_model(self, parameters) -> NeuralNetwork:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    # This is where actions the agent take are calculated, fitness is modified here.
    def run_single(self, env, logger, render=False) -> Tuple[float, np.array]:
        done = False
        state = env.reset()
        fitness = 0
        old_fitness = None
        agent_parameters = self.nn.agent_parameters
        n_episodes = agent_parameters['n_episodes']
        frames = np.zeros(shape=(32, 4, agent_parameters['downsample_w'], agent_parameters['downsample_h']))
        deadrun = n_episodes / 2
        checkfit = int(round(n_episodes / 3))
        for episode in range(n_episodes):
            logger.tick()

            if render:
                env.render()
            obs = torch.from_numpy(state.copy()).float()
            processed_state = self.nn.preprocess.forward(obs[episode % 4,])
            frames[episode % 4, 0] = processed_state
            data = torch.from_numpy(frames).to(self.nn.device)
            action = self.nn.forward(data)
            actions = np.array(action.cpu().detach().numpy())

            action = get_weighted_action(actions) % 7
            # Repeat the action for a few frames
            for _ in range(4):
                obs, reward, done, _ = env.step(action)
                fitness += reward
                if done:
                    break

            # If the fitness isn't improved much at the start, it's likely it got stuck standing still or something
            # so just kill it then and there, this could be better checked by comparing previous results from like
            # a few episodes ago
            if episode > checkfit:
                old_fitness = fitness

            if episode > deadrun and fitness <= old_fitness:
                break

            if done:
                break

        return fitness, self.nn.get_weights_biases()
