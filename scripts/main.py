import copy
import math
import sys
import time

import gym
import numpy as np
from matplotlib import pyplot as plt
import concurrent.futures
from environment import MarioEnvironment
from environment.util import LoadingLog
from ga.components.Individuals import CNNIndividual, ReinforcementCNNIndividual
from ga.components.Population import Population
from ga.util import MarioGAUtil
from ga.util.ReplayMemory import ReplayMemory
from nn.setup import AgentParameters


def run_single_wrapper(level, i, num_threads, run_batch):
    agents_to_load_count = len(agents_for_level)

    def run_single_with_index(index):
        fitness, _ = agents_for_level[index % agents_to_load_count].run_single([level], None, render=False, index=index)
        return fitness

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(run_single_with_index, range(run_batch)))

    average_fitness = sum(results) / run_batch
    agent_x.append(i)
    agent_y.append(average_fitness)
    print("level " + level + " is " + str(average_fitness))

levels = ["SuperMarioBros-1-1-v0", "SuperMarioBros-2-1-v0", "SuperMarioBros-3-1-v0", "SuperMarioBros-4-1-v0"]

param = AgentParameters.MarioCudaAgent().agent_parameters
replay_memory = ReplayMemory(param['memory_size'])
logger = LoadingLog.PrintLoader(param.get('experience_episodes'), 'x')
# Ignore 'generation' in the print logger, it's just the same agent running multiple times
agent_x = []  # Level
agent_y = []  # Average Fitness
run_batch = 100
agents_to_load = ["seed_agents/Mario_1_1_9950000", "seed_agents/Mario_2_1_9925000", "seed_agents/Mario_3_1_10000000",
                  "seed_agents/Mario_4_1_9950000"]
agents_for_level = []
train_time = '10m'

for agent_name in agents_to_load:
    agent = CNNIndividual(AgentParameters.MarioCudaAgent().agent_parameters, replay_memory)
    agent.nn.load('../models/' + agent_name + '.npy')
    agents_for_level.append(agent)

for i, level in enumerate(levels):
    run_single_wrapper(level, i, 100, run_batch)

level_names = ['World 1-1', 'World 2-1', 'World 3-1', 'World 4-1']
plt.figure(figsize=(10, 6))  # Optional: Set the figure size

plt.bar(agent_x, agent_y, color='skyblue')
plt.xlabel('Level')  # X-axis label
plt.ylabel('Average Fitness')  # Y-axis label
plt.title(
    f'Average World Progression for Seed Agent on specialized levels')  # Attempted Merge Agent Train Time {train_time}')
plt.xticks(range(len(agent_x)), level_names, rotation=45)
plt.axhline(35, color='red', linestyle='--', label=f'Other World Level Flags')
plt.axhline(30, color='blue', linestyle='--', label=f'3-1 Level Flag')
plt.legend()
plt.tight_layout()  # Optional: Ensure the labels fit within the plot area

plt.show()
