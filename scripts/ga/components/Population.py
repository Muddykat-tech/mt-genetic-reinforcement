import copy
from datetime import datetime
from typing import Callable

import numpy as np
import torch

from environment.util import LoadingLog
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

        # Setting up the logger
        self.logger = LoadingLog.PrintLoader(self.n_generations, '#')
        self.logger.reset()

    def set_population(self, population: list):
        self.old_population = population

    def update_old_population(self):
        self.old_population = copy.deepcopy(self.new_population)

    def get_best_model_parameters(self) -> np.array:
        return sorted(self.new_population, key=lambda ind: ind.fitness, reverse=True)[0]

    def save_model_parameters(self, output_folder, iterations, save_as_pytorch=False):
        best_model = self.get_best_model_parameters()
        file_name = self.get_file_name(self.now()) + f'_I={iterations}_SCORE={best_model.fitness}.npy'
        output_filename = output_folder + '-' + file_name
        if save_as_pytorch:
            torch.save(best_model.weights_biases, output_filename)
        else:
            np.save(output_filename, best_model.weights_biases)

    def get_file_name(self, date):
        return '{}_NN={}_POPSIZE={}_GEN={}_PMUTATION_{}_PCROSSOVER_{}'.format(date,
                                                                              self.new_population[0].__class__.__name__,
                                                                              self.population_size,
                                                                              self.n_generations,
                                                                              self.p_mutation,
                                                                              self.p_crossover)

    def run(self, env, run_generation: Callable, output_folder=None):
        best_individual = sorted(self.old_population, key=lambda ind: ind.fitness, reverse=True)[0]
        logger = self.logger
        render = self.population_settings['render_mode']
        print('Population Settings: \n' + str(self.population_settings))
        print('Training Model:')
        for i in range(self.n_generations):
            logger.printProgress(i)

            [p.calculate_fitness(env, logger, render) for p in self.old_population]

            self.new_population = [None for _ in range(self.population_size)]
            run_generation(env, self.old_population, self.new_population, self.population_settings, logger)

            self.update_old_population()

            torch.cuda.empty_cache()  # Hopefully this helps with the memory issues

            new_best_individual = self.get_best_model_parameters()

            if new_best_individual.fitness > best_individual.fitness:
                best_individual = new_best_individual

        print('')
        print('Saving best model with fitness: {}'.format(best_individual.fitness))
        self.save_model_parameters(output_folder, 0, save_as_pytorch=False)

    def show_stats(self, n_gen):
        mean, min, max = statistics(self.new_population)
        date = self.now()
        stats = f"{date} - generation {n_gen + 1} | mean: {mean}\tmin: {min}\tmax: {max}\n"
        print('')
        print(stats)

    @staticmethod
    def now():
        return datetime.now().strftime('%m-%d-%Y_%H-%M')
