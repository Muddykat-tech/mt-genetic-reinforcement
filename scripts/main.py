import copy
import math
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

env = MarioEnvironment.create_mario_environment()

# Setup Population Settings for Genetic Algorithm Training. (Move this to a specified settings script)
population_settings = {}

population_settings['agent-reinforcement'] = [2, ReinforcementCNNIndividual,
                                              AgentParameters.MarioCudaAgent().agent_parameters]
population_settings['agent-generic'] = [2, CNNIndividual, AgentParameters.MarioCudaAgent().agent_parameters]
population_settings['p_mutation'] = 0.05
population_settings['p_crossover'] = 0.8
population_settings['n_generations'] = 25
population_settings['render_mode'] = 0

param = AgentParameters.MarioCudaAgent()
replay_memory = ReplayMemory(param.agent_parameters['memory_size'])

population = Population(population_settings, replay_memory)
population.run(env, MarioGAUtil.generation, '../../models/')

# Run an agent directly, change it's settings in Agent Parameters.MarioCudaAgent()
# TODO make a separate param for agents
param = AgentParameters.MarioCudaAgent().agent_parameters
logger = LoadingLog.PrintLoader(param.get('experience_episodes'), 'x')
agent = CNNIndividual(AgentParameters.MarioCudaAgent().agent_parameters, replay_memory)
# Ignore 'generation' in the print logger, it's just the same agent running multiple times
agent_x = []  # Timestep
agent_y = []  # Fitness

model_name = 'MERGED-10-09-2023_17-53_NN=CNNIndividual_POPSIZE=32_GEN=25_PMUTATION_0.05_PCROSSOVER_0.8_BATCH_SIZE=32__I=0_SCORE=35.49999999999995'
agent.nn.load('../models/' + model_name + '.npy')

agent.run_single(env, logger, render=True, agent_x=agent_x, agent_y=agent_y)

plt.plot(agent_x, agent_y, color='blue', marker='o')
plt.title('Fitness of Merged Agent in 3-1')
plt.xlabel('Episode')
plt.ylabel('Fitness')
legend_info = f'Time Taken: {logger.get_estimate()}'
plt.legend([legend_info], loc='upper left', fontsize=10)
plt.grid(True)
plt.savefig('../graphs/Merged-' + str(time.time()) + '.png')
#
# file_name = "episodes-" + str(param.get('experience_episodes')) + "-Agent-Fitness-[" + str(agent.fitness) + "].npy"
# output_filename = '../models/' + '-' + file_name
#
# np.save(output_filename, agent.weights_biases)