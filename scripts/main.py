import copy
import math
import sys
import time

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

levels = ["SuperMarioBros-1-1-v0", "SuperMarioBros-2-1-v0", "SuperMarioBros-3-1-v0", "SuperMarioBros-4-1-v0"]

param = AgentParameters.MarioCudaAgent().agent_parameters
replay_memory = ReplayMemory(param['memory_size'])
logger = LoadingLog.PrintLoader(param.get('experience_episodes'), 'x')
agent = CNNIndividual(AgentParameters.MarioCudaAgent().agent_parameters, replay_memory)
# Ignore 'generation' in the print logger, it's just the same agent running multiple times
agent_x = []  # Level
agent_y = []  # Average Fitness

model_name = '-10-13-2023_16-24_NN=CNNIndividual_POPSIZE=28_GEN=5_PMUTATION_0.05_PCROSSOVER_0.8_BATCH_SIZE=32__I=0_SCORE=31.52999999999997'
agent.nn.load('../models/' + model_name + '.npy')

for i in range(4):
    level = levels[i]
    env = MarioEnvironment.create_mario_environment(level)
    agent_fitness = 0

    for x in range(100):
        fitness, _ = agent.run_single(env, None, render=False)
        agent_fitness += fitness

    average_fitness = agent_fitness / 100
    agent_x.append(i)
    agent_y.append(average_fitness)
    print(" level " + level + " is " + str(average_fitness))
    env.close()

level_names = ['World 1-1', 'World 2-1', 'World 3-1', 'World 4-1']
plt.figure(figsize=(10, 6))  # Optional: Set the figure size

plt.bar(agent_x, agent_y, color='skyblue')
plt.xlabel('Level')  # X-axis label
plt.ylabel('Average Fitness')  # Y-axis label
plt.title('Average World Progression for Merge Agent trained 1h')
plt.xticks(range(len(agent_x)), level_names, rotation=45)
plt.axhline(25, color='red', linestyle='--', label=f'Level Flag')
plt.legend()
plt.tight_layout()  # Optional: Ensure the labels fit within the plot area

plt.show()
