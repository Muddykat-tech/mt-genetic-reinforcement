import copy
import math
import time
from multiprocessing import freeze_support

import gym
import numpy as np
from matplotlib import pyplot as plt

from dialog import Dialog
from environment import MarioEnvironment
from environment.util import LoadingLog
from ga.components.Individuals import CNNIndividual, ReinforcementCNNIndividual
from ga.components.Population import Population
from ga.util import MarioGAUtil, Holder
from ga.util.ReplayMemory import ReplayMemory
from nn.setup import AgentParameters

if __name__ == '__main__':
    freeze_support()

    levels = ["SuperMarioBros-1-1-v0"]  # , "SuperMarioBros-2-1-v0", "SuperMarioBros-3-1-v0", "SuperMarioBros-4-1-v0"]
    levels = levels * 1

    agents = ["Mario_1_1_9950000", "Mario_2_1_9925000", "Mario_3_1_10000000", "Mario_4_1_9950000"]
    agents = agents * 0

    # Setup Population Settings for Genetic Algorithm Training. (Move this to a specified settings script)
    population_settings = {}

    population_settings['agent-reinforcement'] = [0, ReinforcementCNNIndividual,
                                                  AgentParameters.MarioCudaAgent().agent_parameters]
    population_settings['agent-generic'] = [100, CNNIndividual, AgentParameters.MarioCudaAgent().agent_parameters]
    population_settings['seed-agents'] = agents
    population_settings['p_mutation'] = 0.02
    population_settings['p_crossover'] = 0.8
    population_settings['n_generations'] = 14
    population_settings['render_mode'] = 0
    population_settings['use_multithreading'] = 1
    population_settings['n_threads'] = 100

    population = Population(population_settings)
    population.run(levels, MarioGAUtil.generation, '../../models/')

    # logger = LoadingLog.PrintLoader(99999, '=')
    # rl_agent = ReinforcementCNNIndividual(AgentParameters.MarioCudaAgent().agent_parameters)
    # rl_agent.run_single(levels, logger, False)

    if len(Holder.fitness_memory) > 0:
        mem_plot = Holder.fitness_memory
        plt.plot(Holder.fitness_memory_ticks, mem_plot, color='blue', marker=',')
        plt.title('Fitness of DQN Agent')
        plt.xlabel('Step')
        plt.ylabel('Fitness')
        legend_info = f'Total Population Steps: {200000}\nFitness: {max(mem_plot)}'
        plt.legend([legend_info], loc='upper left', fontsize=10)
        plt.grid(True)
        plt.savefig('../../new-graph/DQN-Fitness-Plot.png')
        plt.close()

        mem_plot = mem_plot[:round(len(mem_plot) / 2)]
        Holder.fitness_memory_ticks = Holder.fitness_memory_ticks[:round(len(Holder.fitness_memory_ticks) / 2)]

        plt.plot(Holder.fitness_memory_ticks, mem_plot, color='blue', marker=',')
        plt.title('Fitness of DQN Agent')
        plt.xlabel('Step')
        plt.ylabel('Fitness')
        legend_info = f'Total Population Steps: {100000}\nFitness: {max(mem_plot)}'
        plt.legend([legend_info], loc='upper left', fontsize=10)
        plt.grid(True)
        plt.savefig('../../new-graph/DQN-Fitness-Plot-100k.png')
        plt.close()

