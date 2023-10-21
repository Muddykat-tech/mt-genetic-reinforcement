import copy
import math
import time
from multiprocessing import freeze_support

import gym
import numpy as np
from matplotlib import pyplot as plt

from environment import MarioEnvironment
from environment.util import LoadingLog
from ga.components.Individuals import CNNIndividual, ReinforcementCNNIndividual
from ga.components.Population import Population
from ga.util import MarioGAUtil
from ga.util.ReplayMemory import ReplayMemory
from nn.setup import AgentParameters

if __name__ == '__main__':
    freeze_support()

    levels = ["SuperMarioBros-1-1-v0", "SuperMarioBros-2-1-v0", "SuperMarioBros-3-1-v0", "SuperMarioBros-4-1-v0"]
    levels = levels * 1

    agents = ["Mario_1_1_9950000", "Mario_2_1_9925000", "Mario_3_1_10000000", "Mario_4_1_9950000"]
    agents = agents * 0

    # Setup Population Settings for Genetic Algorithm Training. (Move this to a specified settings script)
    population_settings = {}

    population_settings['agent-reinforcement'] = [2, ReinforcementCNNIndividual,
                                                  AgentParameters.MarioCudaAgent().agent_parameters]
    population_settings['agent-generic'] = [0, CNNIndividual, AgentParameters.MarioCudaAgent().agent_parameters]
    population_settings['seed-agents'] = agents
    population_settings['p_mutation'] = 0.05
    population_settings['p_crossover'] = 0.8
    population_settings['n_generations'] = 500
    population_settings['render_mode'] = 0
    population_settings['use_multithreading'] = 1
    population_settings['n_threads'] = 16

    param = AgentParameters.MarioCudaAgent()
    replay_memory = ReplayMemory(param.agent_parameters['memory_size'])

    population = Population(population_settings, replay_memory)
    population.run(levels, MarioGAUtil.generation, '../../models/')