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
# Ignore 'generation' in the print logger, it's just the same agent running multiple times
agent_x = []  # Level
agent_y = []  # Average Fitness
run_batch = 10
agents_to_load = ["-10-18-2023_17-45_NN=CNNIndividual_POPSIZE=4_GEN=2_PMUTATION_0.05_PCROSSOVER_0.8_BATCH_SIZE=32__I=0_SCORE=158.90999999999957"]
agents_for_level = []

for agent_name in agents_to_load:
    agent = CNNIndividual(AgentParameters.MarioCudaAgent().agent_parameters, replay_memory)
    agent.nn.load('../models/' + agent_name + '.npy')
    agents_for_level.append(agent)

for i in range(4):
    level = levels[i]
    agent_fitness = 0

    for x in range(run_batch):
        fitness, _ = agents_for_level[i % len(agents_to_load)].run_single([level], None, render=False)
        agent_fitness += fitness

    average_fitness = agent_fitness / run_batch
    agent_x.append(i)
    agent_y.append(average_fitness)
    print(" level " + level + " is " + str(average_fitness))

level_names = ['World 1-1', 'World 2-1', 'World 3-1', 'World 4-1']
plt.figure(figsize=(10, 6))  # Optional: Set the figure size

plt.bar(agent_x, agent_y, color='skyblue')
plt.xlabel('Level')  # X-axis label
plt.ylabel('Average Fitness')  # Y-axis label
plt.title('Average World Progression for Attempted Merge Agent Train Time 45m')
plt.xticks(range(len(agent_x)), level_names, rotation=45)
plt.axhline(30, color='red', linestyle='--', label=f'Level Flag')
plt.legend()
plt.tight_layout()  # Optional: Ensure the labels fit within the plot area

plt.show()
