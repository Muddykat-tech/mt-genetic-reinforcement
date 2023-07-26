import copy
from datetime import datetime
from typing import Callable

import numpy as np
import torch


class Population:
    def __init__(self, individuals: list, n_generations: int, p_mutation: float, p_crossover: float):
        # Individuals is a list of Individual Class NN
        self.old_population = [copy.copy(individual) for individual in individuals]
        self.new_population = []
        self.population_size = len(individuals)

        # Each individual should handle its own episode amount in the AgentParameter Settings
        # Setting hyperparameters for ga:
        self.p_mutation = p_mutation
        self.n_generations = n_generations
        self.p_crossover = p_crossover

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

    def run(self, env, run_generation: Callable, verbose=False, log=False, output_folder=None):
        best_individual = sorted(self.old_population, key=lambda ind: ind.fitness, reverse=True)[0]
        for i in range(self.n_generations):
            [p.calculate_fitness(env, p.get("n_episodes")) for p in self.old_population]

            self.new_population = [None for _ in range(self.population_size)]
            run_generation(env, self.old_population, self.new_population, self.p_mutation, self.p_crossover)

            self.update_old_population()

            new_best_individual = self.get_best_model_parameters()

            if new_best_individual > best_individual.fitness:
                print('Saving new best model with fitness: {}'.format(new_best_individual.fitness))
                self.save_model_parameters(output_folder, i, False)
                best_individual = new_best_individual


    @staticmethod
    def now():
        return datetime.now().strftime('%m-%d-%Y_%H-%M')