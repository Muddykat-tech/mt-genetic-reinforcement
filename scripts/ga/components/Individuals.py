import copy
import math
import random
import statistics
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import namedtuple, deque
from itertools import count
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim

from environment import MarioEnvironment
from ga.util import Holder
from ga.util.ReplayMemory import Transition
from nn.agents.CNN import CNN
from nn.preprocess import preproc
import datetime

lock = threading.Lock


def compute_weighted_fitness(values, similarity_weight, score_weight):
    n = len(values)
    sum_of_values = sum(values)
    abs_diff_sum = 0

    for i in range(n):
        for j in range(n):
            abs_diff_sum += abs(values[i] - values[j])

    x = (1/n) * (similarity_weight * (sum_of_values - abs_diff_sum) + score_weight * sum_of_values)  # Normalize the equation
    return x


# Abstract Individual Class that has base functions the GA can call upon
class Individual(ABC):
    def __init__(self, parameters):
        self.nn = self.get_model(parameters)
        self.fitness = 0.0
        self.steps_done = 0
        self.weights_biases: np.array = None

    def calculate_fitness(self, levels, logger, render, index, generation) -> None:
        self.fitness, self.weights_biases, self.steps_done = self.run_single(levels, logger, render, index=index,
                                                                             generation=generation)

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)

    @abstractmethod
    def get_model(self, parameters) -> CNN:
        pass

    @abstractmethod
    def run_single(self, levels, logger, render=False, index=0, generation=0) -> Tuple[
        float, np.array, int]:
        pass


# Convolutional Neural Network Individual

class CNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.estimate = 'NA'
        self.replay_memory = Holder.replay_memory
        self.level_fitness = {}

    def get_model(self, parameters) -> CNN:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    def shannon_entropy(self, probabilities):
        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy

    # This is where actions the agent take are calculated, fitness is modified here.
    def run_single(self, levels, logger, render=False, index=0, generation=0) -> Tuple[
        float, np.array, int]:
        steps_done = 0
        loading_progress = ['■ □ □', '□ ■ □', '□ □ ■']
        levels_to_run = len(levels)
        fitness = 0
        self.level_fitness = {}
        for selected_level in range(levels_to_run):
            fitness = 0
            env = MarioEnvironment.create_mario_environment(levels[selected_level])
            global next_state
            state, old_info = env.reset()
            agent_parameters = self.nn.agent_parameters
            n_episodes = agent_parameters['n_episodes']
            self.nn.to(self.nn.device)
            for episode in range(n_episodes):
                if logger is not None:
                    logger.print(f'Calculating Individual {index}')
                    logger.tick()
                    pass
                else:
                    update = int(time.time()) % len(loading_progress)
                    sys.stdout.write(
                        "\r | Calculating Fitness {Agent[" + str(index) + "]} | " + loading_progress[update] + " |")
                    sys.stdout.flush()

                if render:
                    if logger is not None:
                        logger.print('Known issue using threads and environmental rendering, gets stuck or crashes')
                        logger.tick()
                    env.render()

                state = state.to(self.nn.device)

                # Determine the action
                action_probability = torch.nn.functional.softmax(
                    self.nn.forward(state).mul(agent_parameters['action_conf']),
                    dim=1)

                m = torch.distributions.Categorical(action_probability)
                action = m.sample().item()
                next_state, reward, done, info = env.step(action)
                steps_done += 1
                reward = min(agent_parameters['reward_max_x_change'], max(-agent_parameters['reward_max_x_change'],
                                                                          info['x_pos'] - old_info[
                                                                              'x_pos']))  # Clip the x_pos difference to deal with warp points, etc.
                old_info = info
                reward /= 100  # 15
                fitness += reward
                self.fitness = fitness
                # Format the generic agent data to ensure it's compatible with Reinforcement Agents' memory
                # if self.replay_memory is not None:
                #     reward = torch.tensor([reward])
                #     action = torch.tensor([[action]], device=self.nn.device, dtype=torch.long)
                #
                #     self.replay_memory.push(state, action, next_state, reward, not done, not info['flag_get'])

                state = next_state
                if done:
                    break

            env.close()
            self.level_fitness[levels[selected_level]] = fitness
            if logger is not None:
                self.estimate = str(logger.get_estimate())

        self.nn.to(torch.device('cpu'))

        # retrieve the fitness score for the different levels
        # values = list(self.level_fitness.values())
        # w1 = 0.25
        # w2 = 0.75
        # self.fitness = compute_weighted_fitness(values, w1, w2)
        self.fitness = fitness
        self.steps_done += steps_done
        return self.fitness, self.nn.get_weights_biases(), self.steps_done


# Convolutional Neural Network Individual
class ReinforcementCNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.temp = True
        self.fitness = 0.0
        self.estimate = 'NA'
        self.replay_memory = Holder.replay_memory
        self.target_nn = self.get_model(parameters)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=0.0000625, betas=(0.9, 0.999), eps=0.00015)
        self.frames = np.zeros(
            shape=(1, parameters['n_frames'], parameters['downsample_w'], parameters['downsample_h']))

    def get_model(self, parameters) -> CNN:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    def select_action(self, env, state, steps_done):

        sample = random.random()

        eps_threshold = self.get('ep_end') + np.maximum(0, (self.get('ep_start') - self.get('ep_end')) * \
                                                        (self.get('ep_decay') - np.maximum(0, steps_done - self.get(
                                                            'learn_start'))) / self.get('ep_decay'))

        # Early in training it's a bit inefficient to do a forward pass if eps_threshold isn't exceeded,
        # but it's useful to see the Q-value output for debugging purposes.
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            net_out = self.nn(state).cpu().numpy()[0]
            greedy_act = np.argmax(net_out)
            max_q = net_out[greedy_act]

            # if self.get('q_val_plot_freq') > 0 and steps_done % self.get('q_val_plot_freq') == 0:
            #     self.q_values_plot.add_data_point("bestq", steps_done, [max_q], True, True)
            #     self.q_values_plot.update_image('eps_threshold = ' + str(eps_threshold))

        steps_done += 1
        memsize = len(self.replay_memory.memory)
        Holder.memory_buffer_history.append(memsize)

        if sample > eps_threshold:
            return greedy_act, steps_done
        else:
            return env.action_space.sample(), steps_done

    def train_step(self):
        batch_size = self.get('batch_size')
        if len(self.replay_memory) < batch_size:
            return

        transitions = self.replay_memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(batch.non_final).to(self.nn.device)
        not_got_flag_mask = torch.tensor(batch.not_got_flag).to(self.nn.device)
        state_batch = torch.cat(batch.state).to(self.nn.device)
        next_states = torch.cat(batch.next_state).to(self.nn.device)
        action_batch = torch.tensor(batch.action).to(self.nn.device).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).to(self.nn.device)

        # Generates the first image of all states in the state buffer
        # if self.temp:
        #     preproc.generate_image(84, 84, state_batch[3][0], 0)
        #     preproc.generate_image(84, 84, state_batch[3][1], 1)
        #     preproc.generate_image(84, 84, state_batch[3][2], 2)
        #     preproc.generate_image(84, 84, state_batch[3][3], 3)
        #     self.temp = False

        state_action_values = self.nn(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
            next_state_values = self.target_nn(next_states).max(1)[0].mul(non_final_mask.long())

        expected_state_action_values = (next_state_values * self.get('gamma') + reward_batch)

        # Don't treat reaching the flag as terminal
        expected_state_action_values = expected_state_action_values.mul(not_got_flag_mask.long()).add(
            torch.ones_like(not_got_flag_mask.long()).sub(not_got_flag_mask.long()).mul(state_action_values))

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_single(self, levels, logger, render=False, index=0, generation=0) -> Tuple[
        float, np.array, int]:
        global next_state
        fitness = 0.0
        old_fitness = 0.0
        steps_done = 0
        moving_fitness = 0
        moving_fitness_mom = 0.95
        moving_fitness_updates = 0
        n_episodes = self.get('n_episodes')
        xp_episodes = self.get('experience_episodes')
        end_training = False

        levels_to_run = len(levels)
        for selected_level in range(levels_to_run):
            level = levels[selected_level]
            env = MarioEnvironment.create_mario_environment(level)
            state, info = env.reset()
            self.nn.to(self.nn.device)
            self.target_nn.to(self.nn.device)
            # Average the fitness result between '
            for episode in range(xp_episodes):  # 'Episodes'
                state, old_info = env.reset()
                old_fitness = self.fitness if old_fitness < self.fitness else old_fitness
                fitness = 0.0
                logger.print(str(episode / xp_episodes) + ' Agent: (R) | Approx: ' + self.estimate)
                if end_training:
                    break
                for t in range(n_episodes):  # rename to n_steps
                    if end_training:
                        break

                    logger.tick()
                    logger.print(str(episode / xp_episodes) + ' Agent: (R) | Approx: ' + self.estimate)
                    if render:
                        env.render()

                    action, steps_done = self.select_action(env, state, steps_done)
                    next_state, reward, done, info = env.step(action)
                    reward = min(self.get('reward_max_x_change'), max(-self.get('reward_max_x_change'),
                                                                      info['x_pos'] - old_info[
                                                                          'x_pos']))  # Clip the x_pos difference to deal with warp points, etc.
                    old_info = info
                    reward /= 100  # 15

                    fitness += reward
                    self.fitness = fitness

                    reward = torch.tensor([reward])

                    self.replay_memory.push(state, action, next_state, reward, not done, not info['flag_get'])

                    state = next_state

                    if steps_done >= self.get('learn_start') and steps_done % self.get('train_freq') == 0:
                        self.train_step()

                    if steps_done % self.nn.agent_parameters['target_update_freq'] == 0:
                        self.target_nn.load_state_dict(self.nn.state_dict())

                    if steps_done % self.get('save_freq') == 0:
                        # if self.get('q_val_plot_freq') > 0:
                        #     self.q_values_plot.save_image(self.get('log_dir') + '/graphs/')
                        # self.fitness_plot.save_image(self.get('log_dir') + '/graphs/')
                        output_filename = self.get('log_dir') + '/models/RL_agent_' + datetime.datetime.now().strftime(
                            "%m%d%Y_%H_%M_%S") + '_' + str(steps_done) + '.npy'
                        np.save(output_filename, self.nn.get_weights_biases())

                    if steps_done >= 200000:
                        # if self.get('q_val_plot_freq') > 0:
                        #     self.q_values_plot.save_image(self.get('log_dir') + '/graphs/')
                        # self.fitness_plot.save_image(self.get('log_dir') + '/graphs/')
                        output_filename = self.get('log_dir') + '/models/RL_agent_' + datetime.datetime.now().strftime(
                            "%m%d%Y_%H_%M_%S") + '_' + str(steps_done) + '.npy'
                        np.save(output_filename, self.nn.get_weights_biases())
                        end_training = True
                        break

                    if done:
                        break

                # Plot smoothed running average of fitness
                moving_fitness = moving_fitness_mom * moving_fitness + (1.0 - moving_fitness_mom) * self.fitness
                moving_fitness_updates += 1
                zero_debiased_moving_fitness = moving_fitness / (1.0 - moving_fitness_mom ** moving_fitness_updates)
                Holder.fitness_memory.append(zero_debiased_moving_fitness)
                Holder.fitness_memory_ticks.append(steps_done)

                if steps_done % 25000 == 0:
                    print("Milestone: " + str(steps_done / 200000))

                # self.fitness_plot.add_data_point("fitness", steps_done, [zero_debiased_moving_fitness], False, True)
                # self.fitness_plot.append(zero_debiased_moving_fitness)

                # Reset the estimate after we've finished one training cycle
            env.close()
        # Help clear GPU memory, by moving network back to the cpu once training run is complete
        self.nn.to(torch.device('cpu'))
        self.target_nn.to(torch.device('cpu'))

        self.fitness = fitness
        self.steps_done += steps_done
        # why is the holder needed? I have no fucking clue
        # But for some reason without it, I get false readings in the Population class, so instead of treating the
        # self.replay_memory as a pointer to a list, I store it in HOLDER and reference it directly

        Holder.memory_buffer_history.append(len(self.replay_memory.memory))
        return fitness, self.nn.get_weights_biases(), self.steps_done
