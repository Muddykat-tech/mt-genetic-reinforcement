# Methods
import copy
import random
from collections import deque, namedtuple
from typing import List, Tuple

import numpy as np

from environment.util import LoadingLog
from ga.components.Individuals import Individual


def crossover(parent1_weights_biases: np.array, parent2_weights_biases: np.array, p: float):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)

    if np.random.rand() < p:
        child1_weights_biases[position:], child2_weights_biases[position:] = \
            child2_weights_biases[position:], child1_weights_biases[position:]
    return child1_weights_biases, child2_weights_biases


def mutation(parent_weights_biases: np.array, p: float):
    child_weight_biases = np.copy(parent_weights_biases)
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        n = np.random.uniform(np.min(child_weight_biases), np.max(child_weight_biases))
        child_weight_biases[position] = n + np.random.randint(-20, 20)
    return child_weight_biases


def ranking_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    sorted_population = sorted(population, key=lambda individual: individual.fitness if individual is not None else 0.0,
                               reverse=True)
    parent1, parent2 = sorted_population[:2]
    return parent1, parent2


def statistics(population: List[Individual]):
    population_fitness = [individual.fitness for individual in population]
    return np.mean(population_fitness), np.min(population_fitness), np.max(population_fitness)


def generation(env, old_population, new_population, p_settings, logger: LoadingLog.PrintLoader):
    for i in range(0, len(old_population) - 1, 2):
        logger.tick()
        p_crossover = p_settings['p_crossover']
        p_mutation = p_settings['p_crossover']
        render_mode = p_settings['render_mode']

        # Selection
        parent1, parent2 = ranking_selection(old_population)

        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 p_crossover)
        # Mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness(env, logger, render_mode)
        child2.calculate_fitness(env, logger, render_mode)

        # If children fitness is greater than the parents update population
        if child1.fitness > parent1.fitness:
            new_population[i] = child1
        else:
            new_population[i] = parent1

        if child2.fitness > parent2.fitness:
            new_population[i + 1] = child2
        else:
            new_population[i + 1] = parent2

    return new_population
