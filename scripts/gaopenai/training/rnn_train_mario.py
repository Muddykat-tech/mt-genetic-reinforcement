import copy
import random
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from collections import OrderedDict, deque

import cv2
import gym
import gym_super_mario_bros
import torch
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from typing import Tuple, Callable, List
from gaopenai.nn.base_nn import NeuralNetwork
from gaopenai.nn.rnn import RNN
import numpy as np
from torch import nn
import torch.nn.functional as F
from gym import spaces
import time

from gaopenai.preprocess.preproc import Preproc
from gaopenai.util.loading import PrintLoader


class NeuralNetwork(ABC):
    @abstractmethod
    def get_weights_biases(self) -> np.array:
        pass

    @abstractmethod
    def update_weights_biases(self, weights_biases: np.array) -> None:
        pass

    def load(self, file):
        self.update_weights_biases(np.load(file))


class RNN(nn.Module, NeuralNetwork):
    def __init__(self, input_size, hidden_size1, output_size):
        super(RNN, self, ).__init__()
        self.fitness = 0.0
        self.hidden_size1 = hidden_size1
        agent_params = {}
        agent_params["gpu"] = 0
        agent_params["use_rgb_for_raw_state"] = True
        agent_params["downsample_w"] = 84
        agent_params["downsample_h"] = 84
        gpu = agent_params["gpu"]

        self.device = torch.device("cuda" if gpu >= 0 else "cpu")
        self.agent_params = agent_params
        self.downsample_w = agent_params["downsample_w"]
        self.downsample_h = agent_params["downsample_h"]
        self.preprocess = Preproc(self.agent_params, self.downsample_w, self.downsample_h)

        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=8, stride=4).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=2, stride=2).to(self.device)
        self.fc1 = nn.Linear(in_features=800, out_features=32).to(self.device)
        self.fc2 = nn.Linear(in_features=32, out_features=output_size).to(self.device)

        self.relu = nn.ReLU().to(self.device)

    def forward(self, raw_input) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.relu(self.conv1(raw_input.float()))
        input = self.relu(self.conv2(input))
        input = input.view(input.size(0), -1)
        fc1 = self.fc1(input)
        input = self.relu(fc1)
        output = self.fc2(input)
        return output

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(self.hidden_size1)

    def get_weights_biases(self) -> np.array:
        parameters = self.state_dict().values()
        parameters = [p.flatten() for p in parameters]
        parameters = torch.cat(parameters, 0)
        return parameters.cpu().detach().numpy()

    def update_weights_biases(self, weights_biases: np.array) -> None:
        weights_biases = torch.from_numpy(weights_biases)
        shapes = [x.shape for x in self.state_dict().values()]
        shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

        partial_split = weights_biases.split(shapes_prod)
        model_weights_biases = []
        for i in range(len(shapes)):
            model_weights_biases.append(partial_split[i].view(shapes[i]))
        state_dict = OrderedDict(zip(self.state_dict().keys(), model_weights_biases))
        self.load_state_dict(state_dict)


class Population:
    def __init__(self, individual, pop_size, max_generation, p_mutation, p_crossover, p_episodes):
        self.p_episodes = p_episodes
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.old_population = [individual for _ in range(pop_size)]
        self.new_population = [None] * pop_size

    def run(self, env, run_generation: Callable, verbose=True, log=False, output_folder=None):
        print("Running Algorithm")
        print("#################")
        loadingLog.reset()
        for i in range(self.max_generation):
            for index, individual in enumerate(self.old_population):
                if (individual != None):
                    is_valid = getattr(individual, "calculate_fitness")
                    if callable(is_valid):
                        individual.calculate_fitness(env, self.p_episodes, index)
                    else:
                        print("Error - Non Fatal: Individual in Population missing core functionality [" + str(
                            type(individual)) + "]")

            run_generation(env, self.old_population, self.new_population, self.p_mutation, self.p_crossover)

            if log:
                self.save_logs(i, output_folder)

            if verbose:
                self.show_stats(i)

            self.update_old_population()
        self.save_model_parameters(output_folder)

    def save_logs(self, n_gen, output_folder):
        """
        CSV format -> date,n_generation,mean,min,max
        """
        date = self.now()
        file_name = 'logs.csv'
        mean, min, max = statistics(self.new_population)
        stats = f'{date},{n_gen},{mean},{min},{max}\n'
        with open(output_folder + file_name, 'a') as f:
            f.write(stats)

    def show_stats(self, n_gen):
        print("")
        mean, min, max = statistics(self.new_population)
        print("")
        date = self.now()
        stats = f"{date} - generation {n_gen + 1} | mean: {mean}\tmin: {min}\tmax: {max}\n"
        print(stats)

    def update_old_population(self):
        self.old_population = copy.deepcopy(self.new_population)

    def save_model_parameters(self, output_folder):
        best_model = self.get_best_model_parameters()
        date = self.now()
        file_name = self.get_file_name(date) + '.npy'
        np.save(output_folder + file_name, best_model)

    def get_best_model_parameters(self) -> np.array:
        """
        :return: Weights and biases of the best individual
        """
        individual = sorted(self.new_population, key=lambda ind: 0.0 if ind is None else ind.fitness, reverse=True)[0]
        return individual.weights_biases

    def get_file_name(self, date):
        return '{}_NN={}_POPSIZE={}_GEN={}_PMUTATION_{}_PCROSSOVER_{}'.format(date,
                                                                              self.new_population[0].__class__.__name__,
                                                                              self.pop_size,
                                                                              self.max_generation,
                                                                              self.p_mutation,
                                                                              self.p_crossover)

    @staticmethod
    def now():
        return datetime.now().strftime('%m-%d-%Y_%H-%M')


class Individual(ABC):
    def __init__(self, input_size, hidden_size, output_size):
        self.nn = self.get_model(input_size, hidden_size, output_size)
        self.fitness = 0.0
        self.weights_biases: np.array = None

    def calculate_fitness(self, env, p_episodes, progress) -> None:
        self.fitness, self.weights_biases = self.run_single(env, p_episodes, False, progress)

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)

    @abstractmethod
    def get_model(self, input_size, hidden_size, hidden_size_2, output_size) -> NeuralNetwork:
        pass

    @abstractmethod
    def run_single(self, env, n_episodes=300, render=False, progress=0) -> Tuple[float, np.array]:
        pass


# Methods
def crossover(parent1_weights_biases: np.array, parent2_weights_biases: np.array, p: float):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)

    if np.random.rand() < p:
        child1_weights_biases[position:], child2_weights_biases[position:] = \
            child2_weights_biases[position:], child1_weights_biases[position:]
    return child1_weights_biases, child2_weights_biases


def mutation(parent_weights_biases: np.array, p: float):
    child_weight_biases = np.copy(parent_weights_biases)
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        n = np.random.uniform(np.min(child_weight_biases), np.max(child_weight_biases))
        child_weight_biases[position] = n + np.random.randint(-20, 20)
    return child_weight_biases


def ranking_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    sorted_population = sorted(population, key=lambda individual: individual.fitness if individual != None else 0.0,
                               reverse=True)
    parent1, parent2 = sorted_population[:2]
    return parent1, parent2


def roulette_wheel_selection(population: List[Individual]):
    total_fitness = np.sum([individual.fitness for individual in population])
    selection_probabilities = [individual.fitness / total_fitness for individual in population]
    pick = np.random.choice(len(population), p=selection_probabilities)
    return population[pick]


def statistics(population: List[Individual]):
    population_fitness = []
    for individual in population:
        if individual != None and individual.fitness is not None:
            population_fitness.append(individual.fitness)
        else:
            print("")
            print("Encountered, Non-Fatal Error - Individual in Population is Null or Fitness is Null: ",
                  str(type(individual)))
            print("Warning, large amounts of these errors will degrade training quality")

    return np.mean(population_fitness), np.min(population_fitness), np.max(population_fitness)


class RNNIndividual(Individual):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.input_size = input_size
        self.fitness = 0.0

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return RNN(input_size, hidden_size, output_size)

    def get_weighted_action(self, action_probabilities):
        action_probabilities = action_probabilities.flatten()
        weighted_indices = []
        for i, val in enumerate(action_probabilities):
            weight = val ** 2  # Adjust the weighting here (e.g., val ** 2, val ** 3)
            weighted_indices.extend([i] * int(weight * 100))  # Scale the weights as desired

        if len(weighted_indices) == 0:
            print("Fatal Error, No Actions Available?")
            return 0
        # Randomly select an index from the weighted indices list
        random_index = random.choice(weighted_indices)
        return random_index

    def run_single(self, env, n_episodes=300, render=False, progress=0) -> Tuple[float, np.array]:
        state = env.reset()
        fitness = 0
        hidden = self.nn.init_hidden()
        # Starting Single Generation
        start = True
        frames = np.zeros(shape=(32, 4, self.nn.downsample_w, self.nn.downsample_h))
        deadrun = n_episodes / 2
        checkfit = int(round(n_episodes / 3))
        for episode in range(n_episodes):
            loadingLog.tick()
            if render:
                env.render()
            obs = torch.from_numpy(state.copy()).float()
            processed_state = self.nn.preprocess.forward(obs[episode % 4,])
            frames[episode % 4, 0] = processed_state

            # test
            data = torch.from_numpy(frames).to(self.nn.device)
            action = self.nn.forward(data)
            actions = np.array(action.cpu().detach().numpy())

            action = self.get_weighted_action(actions) % 7
            loadingLog.printProgress(episode)
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


def generation(env, old_population, new_population, p_mutation, p_crossover):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
        parent1, parent2 = ranking_selection(old_population)

        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 p_crossover)
        # Mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness(env, EPISODE_TIME, progress=i)
        child2.calculate_fitness(env, EPISODE_TIME, progress=(i + 1))

        # If children fitness is greater thant parents update population
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2


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


# Main Run Commands
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = ConcatObs(env=env, k=4)

# Implement a breakout save, to ensure we can save models even if we don't want to train them for the full generations
POPULATION_SIZE = 20
MAX_GENERATIONS = 80
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.66
EPISODE_TIME = 1000

# NN architecture
INPUT_SIZE = 4
HIDDEN_SIZE = 32
OUTPUT_SIZE = 7

loadingLog = PrintLoader(EPISODE_TIME, "#")

# Implement multi individual populations
p = Population(RNNIndividual(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE), POPULATION_SIZE, MAX_GENERATIONS,
               MUTATION_RATE,
               CROSSOVER_RATE, EPISODE_TIME)

p.run(env, generation, verbose=True, output_folder='../../../models/', log=True)

env.close()
