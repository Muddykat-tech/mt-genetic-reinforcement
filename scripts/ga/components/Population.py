import copy
import gc
from datetime import datetime
from functools import reduce
from typing import Callable

import numpy as np
import torch

import matplotlib.pyplot as plt
from environment.util import LoadingLog
from ga.components.Individuals import ReinforcementCNNIndividual
from ga.util.MarioGAUtil import statistics


class Population:
    def __init__(self, population_settings):
        self.population_settings = population_settings
        self.old_population = []

        # Initialize temp population variables
        generic_population = []
        reinforcement_population = []

        # Setup all instances of a generic agent
        generic_agent_settings = population_settings['agent-generic']
        if generic_agent_settings is not None:
            generic_agent = generic_agent_settings[1]
            generic_population = [generic_agent(generic_agent_settings[2]) for _ in range(generic_agent_settings[0])]

        reinforcement_agent_settings = population_settings['agent-reinforcement']
        if reinforcement_agent_settings is not None:
            reinforcement_agent = reinforcement_agent_settings[1]  # Keep agent default settings
            reinforcement_population = [reinforcement_agent(reinforcement_agent_settings[2]) for _ in
                                        range(reinforcement_agent_settings[0])]

        self.old_population = generic_population + reinforcement_population
        self.population_size = len(self.old_population)
        self.new_population = []

        # Setting hyperparameters for ga:
        self.p_mutation = population_settings['p_mutation']
        self.n_generations = population_settings['n_generations']
        self.p_crossover = population_settings['p_crossover']
        self.batch_size = reinforcement_agent_settings[2]['batch_size']

        # Section for Data Logging

        # Setting up the print logger
        self.logger = LoadingLog.PrintLoader(self.n_generations, '#')
        self.logger.reset()

        # Setting up the structure for the output graph
        self.generic_x_axis = []  # Generation No
        self.generic_y_axis = []  # Fitness No
        self.reinforcement_x_axis = []  # Generation No
        self.reinforcement_y_axis = []  # Fitness No

    def set_population(self, population: list):
        self.old_population = population

    def update_old_population(self):
        # Move tensors from GPU to CPU to release GPU memory
        self.old_population = [agent.nn.to(torch.device('cpu')) for agent in self.old_population]
        self.old_population = copy.deepcopy(self.new_population)

        # Release all unused GPU memory cache
        torch.cuda.empty_cache()

        # And manually Trigger the garbage collector
        gc.collect()

    def get_best_model_parameters(self) -> np.array:
        return sorted(self.new_population, key=lambda ind: ind.fitness, reverse=True)[0]

    def save_model_parameters(self, output_folder, iterations, save_as_pytorch=False, specific_model=None):
        best_model = self.get_best_model_parameters() if specific_model is None else specific_model
        file_name = self.get_file_name(self.now()) + f'_I={iterations}_SCORE={best_model.fitness}.npy'
        output_filename = output_folder + '-' + file_name
        if save_as_pytorch:
            torch.save(best_model.weights_biases, output_filename)
        else:
            np.save(output_filename, best_model.weights_biases)

    def get_file_name(self, date):
        return '{}_NN={}_POPSIZE={}_GEN={}_PMUTATION_{}_PCROSSOVER_{}_BATCH_SIZE={}_'.format(date,
                                                                                             self.new_population[
                                                                                                 0].__class__.__name__,
                                                                                             self.population_size,
                                                                                             self.n_generations,
                                                                                             self.p_mutation,
                                                                                             self.p_crossover,
                                                                                             self.batch_size)

    def run(self, env, run_generation: Callable, output_folder=None):
        best_individual = sorted(self.old_population, key=lambda ind: ind.fitness, reverse=True)[0]
        logger = self.logger
        render = self.population_settings['render_mode']
        print('Population Settings: \n' + str(self.population_settings))
        print('Training Model:')
        for i in range(self.n_generations):
            logger.print_progress(i)

            [p.calculate_fitness(env, logger, render) for p in self.old_population]

            self.new_population = [None for _ in range(self.population_size)]
            run_generation(env, self.old_population, self.new_population, self.population_settings, logger)

            self.update_old_population()

            new_best_individual = self.get_best_model_parameters()

            l_avg = reduce(lambda total, agent: total + agent.fitness, self.old_population, 0) / len(self.old_population)

            self.generic_x_axis.append(i)
            self.generic_y_axis.append(l_avg)

            if new_best_individual.fitness > best_individual.fitness:
                best_individual = new_best_individual

        if len(self.generic_x_axis) > 0:
            plt.plot(self.generic_x_axis, self.generic_y_axis, color='red', marker='o')
            plt.title('Best Fitness of the Agents, Population = ' + str(len(self.old_population)))
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.grid(True)
            plt.savefig('../../graphs/Generic-' + self.get_file_name(self.now()) + '.png')

        print('')
        print('Saving best model in current pop')
        self.save_model_parameters(output_folder, 0, save_as_pytorch=False)
        print('Saving best model of all time - with fitness: {}'.format(best_individual.fitness))
        self.save_model_parameters(output_folder, 0, save_as_pytorch=False, specific_model=best_individual)

    def show_stats(self, n_gen):
        mean, min, max = statistics(self.new_population)
        date = self.now()
        stats = f"{date} - generation {n_gen + 1} | mean: {mean}\tmin: {min}\tmax: {max}\n"
        print('')
        print(stats)

    @staticmethod
    def now():
        return datetime.now().strftime('%m-%d-%Y_%H-%M')
