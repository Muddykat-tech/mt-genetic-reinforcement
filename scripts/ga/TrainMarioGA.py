import copy

from environment import MarioEnvironment
from environment.util import LoadingLog
from ga.components.Individuals import CNNIndividual, ReinforcementCNNIndividual
from ga.components.Population import Population
from ga.util import MarioGAUtil
from nn.setup import AgentParameters

env = MarioEnvironment.create_mario_environment()

# Setup Population Settings for Genetic Algorithm Training. (Move this to a specified settings script)
population_settings = {}

population_settings['agent-reinforcement'] = [10, ReinforcementCNNIndividual,
                                              AgentParameters.MarioCudaAgent().agent_parameters]
population_settings['agent-generic'] = [0, CNNIndividual, AgentParameters.MarioCudaAgent().agent_parameters]
population_settings['p_mutation'] = 0.05
population_settings['p_crossover'] = 0.8
population_settings['n_generations'] = 5
population_settings['render_mode'] = 1

population = Population(population_settings)
population.run(env, MarioGAUtil.generation, '../../models/')

# Run an agent directly, change it's settings in Agent Parameters.MarioCudaAgent()
# TODO make a separate param for agents
# logger = LoadingLog.PrintLoader(1, '?')
# agent = ReinforcementCNNIndividual(AgentParameters.MarioCudaAgent().agent_parameters)
# agent.run_single(env, logger, True)
