# Methods
import copy
import random
import threading
from collections import deque, namedtuple
from typing import List, Tuple

import numpy as np

from environment.util import LoadingLog
from ga.components.Individuals import Individual, CNNIndividual, ReinforcementCNNIndividual
from concurrent.futures import ThreadPoolExecutor, wait

from concurrent.futures import ThreadPoolExecutor, wait

lock = threading.Lock()
dqn_agent = None


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


# return the 'elite' of the population in a separate list
def elitism(population: List[Individual]) -> List[Individual]:
    e_percent = 0.10
    n_elite = int(len(population) * e_percent)

    population.sort(key=lambda individual: individual.fitness, reverse=True)

    elite_individuals = population[:n_elite]

    return elite_individuals


def tournament_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    # Select 2 potential parents
    pop_percent = int(len(population) / 2)
    parents = random.sample(population, k=pop_percent)
    # Get the best two of the random selection
    parents = sorted(parents, key=lambda agent: agent.fitness, reverse=True)

    # Return them
    return parents[0], parents[1]


def statistics(population: List[Individual]):
    population_fitness = [individual.fitness for individual in population]
    return np.mean(population_fitness), np.min(population_fitness), np.max(population_fitness)


def process_individual(i, old_population, new_population, p_settings, logger, levels, generation_id):
    logger.tick()
    p_crossover = p_settings['p_crossover']
    p_mutation = p_settings['p_crossover']
    render_mode = p_settings['render_mode']

    # Selection
    parent1, parent2 = tournament_selection(old_population)
    max_try = 10
    for _ in range(max_try):
        if isinstance(parent1, ReinforcementCNNIndividual) and isinstance(parent2, ReinforcementCNNIndividual):
            parent1, parent2 = tournament_selection(old_population)

    # Crossover
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases, parent2.weights_biases,
                                                             p_crossover)
    # Mutation
    child1.weights_biases = mutation(child1.weights_biases, p_mutation)
    child2.weights_biases = mutation(child2.weights_biases, p_mutation)

    # Update model weights and biases
    child1.update_model()
    child2.update_model()

    child1.steps_done = parent1.steps_done
    child2.steps_done = parent2.steps_done

    if not isinstance(child1, ReinforcementCNNIndividual):
        child1.calculate_fitness(levels, logger, render_mode, i)

    if not isinstance(child2, ReinforcementCNNIndividual):
        child2.calculate_fitness(levels, logger, render_mode, i + 1)

    with lock:
        # If children fitness is greater than the parents, update population
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2


def generation(levels, old_population, new_population, p_settings, logger: LoadingLog.PrintLoader, use_multithreading,
               generation_id):
    global dqn_agent

    elite_population = []  # elitism(old_population)

    for obj in old_population:
        if isinstance(obj, ReinforcementCNNIndividual):
            dqn_agent = obj

    if use_multithreading:
        num_threads = p_settings['n_threads']

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(0, len(old_population) - 1, 2):
                future = executor.submit(process_individual, i, old_population, new_population, p_settings, logger,
                                         levels, generation_id)
                futures.append(future)

            # Wait for all threads to finish
            wait(futures)
    else:
        for i in range(0, len(old_population) - 1, 2):
            process_individual(i, old_population, new_population, p_settings, logger, levels, generation_id)

    # if len(elite_population) > 0:
    #     new_population = sorted(new_population, key=lambda agent: agent.fitness, reverse=True)
    #     new_population = new_population[:-len(elite_population)]
    #     refined_population = new_population + elite_population
    #     refined_population[-1] = dqn_agent
    #     return refined_population

    return new_population

# Changed Algorithm to match the solution proposed in https://youtu.be/ziMHaGQJuSI?si=ijzXnefsGfgVaAxx
