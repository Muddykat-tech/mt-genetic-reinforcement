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
        frames = np.zeros(shape=(1, agent_parameters['n_frames'], agent_parameters['downsample_w'], agent_parameters['downsample_h']))
        deadrun = n_episodes / 2
        checkfit = int(round(n_episodes / 3))
        for episode in range(n_episodes):
            logger.tick()

            if render:
                env.render()
            obs = torch.from_numpy(state.copy()).float()

            # Update frames
            processed_state = self.nn.preprocess.forward(obs[episode % agent_parameters['n_frames'], ])
            frames[0, episode % agent_parameters['n_frames']] = processed_state
            data = torch.from_numpy(frames).to(self.nn.device)

            # Determine the action
            action_probability = torch.nn.functional.softmax(self.nn.forward(data).mul(agent_parameters['action_conf']), dim=1)
            m = torch.distributions.Categorical(action_probability)
            action = m.sample().item()

            # Repeat the action for a few frames
            for _ in range(agent_parameters['n_repeat']):
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


# Convolutional Neural Network Individual
class ReinforcementCNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)

    def get_model(self, parameters) -> NeuralNetwork:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    # This is where actions the agent take are calculated, fitness is modified here.
    def run_single(self, env, logger, render=False) -> Tuple[float, np.array]:
        fitness = 0.0
        discount_factor = 0.95
        eps = 0.5
        eps_decay_factor = 0.999
        model = self.nn
        state = env.reset()
        eps *= eps_decay_factor
        done = False
        while not done:
            if np.random.random() < eps:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(
                    model.forward(np.identity(env.observation_space.n)[state:state + 1]))

            new_state, reward, done, _ = env.step(action)
            target = reward + discount_factor * np.max(
                    model.forward(
                        np.identity(env.observation_space.shape)[new_state:new_state + 1]))
            target_vector = model.forward(
                np.identity(env.observation_space.shape)[state:state + 1])[0]
            target_vector[action] = target
            state = new_state

            fitness += target

        return fitness, self.nn.get_weights_biases()