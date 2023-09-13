import copy
import math
import time

from matplotlib import pyplot as plt

from environment import MarioEnvironment
from environment.util import LoadingLog
from ga.components.Individuals import CNNIndividual, ReinforcementCNNIndividual
from ga.components.Population import Population
from ga.util import MarioGAUtil
from nn.setup import AgentParameters

env = MarioEnvironment.create_mario_environment_random()

# Setup Population Settings for Genetic Algorithm Training. (Move this to a specified settings script)
population_settings = {}

population_settings['agent-reinforcement'] = [4, ReinforcementCNNIndividual,
                                              AgentParameters.MarioCudaAgent().agent_parameters]
population_settings['agent-generic'] = [0, CNNIndividual, AgentParameters.MarioCudaAgent().agent_parameters]
population_settings['p_mutation'] = 0.05
population_settings['p_crossover'] = 0.8
population_settings['n_generations'] = 25
population_settings['render_mode'] = 0

# population = Population(population_settings)
# population.run(env, MarioGAUtil.generation, '../../models/')

# Run an agent directly, change it's settings in Agent Parameters.MarioCudaAgent()
# TODO make a separate param for agents
param = AgentParameters.MarioCudaAgent().agent_parameters
logger = LoadingLog.PrintLoader(param.get('experience_episodes'), 'x')
agent = ReinforcementCNNIndividual(AgentParameters.MarioCudaAgent().agent_parameters)
# Ignore 'generation' in the print logger, it's just the same agent running multiple times
agent_x = []  # Timestep
agent_y = []  # Fitness
agent.run_single(env, logger, True, agent_x, agent_y)
plt.plot(agent_x, agent_y, color='blue', marker='o')
plt.title('Fitness of RL Agent Over Time')
plt.xlabel('Episode')
plt.ylabel('Fitness')
legend_info = f'Time Taken: {logger.get_estimate()}'
plt.legend([legend_info], loc='upper left', fontsize=10)
plt.grid(True)
plt.savefig('../../graphs/RL-' + str(time.time()) + '.png')