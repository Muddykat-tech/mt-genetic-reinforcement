import copy

from environment import MarioEnvironment
from ga.components.Individuals import CNNIndividual, ReinforcementCNNIndividual
from ga.components.Population import Population
from ga.util import MarioGAUtil
from nn.setup import AgentParameters

env = MarioEnvironment.create_mario_environment()

# Setup Population Settings for Genetic Algorithm Training. (Move this to a specified settings script)
population_settings = {}

population_settings['agent-reinforcement'] = [2, ReinforcementCNNIndividual, AgentParameters.MarioCudaAgent().agent_parameters]
population_settings['agent-generic'] = [22, CNNIndividual, AgentParameters.MarioCudaAgent().agent_parameters]
population_settings['p_mutation'] = 0.33
population_settings['p_crossover'] = 0.5
population_settings['n_generations'] = 50
population_settings['render_mode'] = 0

population = Population(population_settings)
population.run(env, MarioGAUtil.generation, '../../models/')