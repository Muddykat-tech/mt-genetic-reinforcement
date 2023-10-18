import copy
import math
import random
import sys
import time
from abc import ABC, abstractmethod
from collections import namedtuple, deque
from itertools import count
from typing import Tuple

import numpy as np
import torch
from torch import optim

from environment import MarioEnvironment
from ga.util.ReplayMemory import Transition
from nn.agents.CNN import CNN
from nn.preprocess import preproc
from dialog import Dialog
from datetime import datetime


# Abstract Individual Class that has base functions the GA can call upon
class Individual(ABC):
    def __init__(self, parameters):
        self.nn = self.get_model(parameters)
        self.fitness = 0.0
        self.weights_biases: np.array = None

    def calculate_fitness(self, levels, logger, render, index) -> None:
        self.fitness, self.weights_biases = self.run_single(levels, logger, render, index=index)

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)

    @abstractmethod
    def get_model(self, parameters) -> CNN:
        pass

    @abstractmethod
    def run_single(self, levels, logger, render=False, agent_x=None, agent_y=None, index=0) -> Tuple[float, np.array]:
        pass


# Convolutional Neural Network Individual
class CNNIndividual(Individual):
    def __init__(self, parameters, replay_memory):
        super().__init__(parameters)
        self.estimate = 'NA'
        self.replay_memory = replay_memory

    def get_model(self, parameters) -> CNN:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    # This is where actions the agent take are calculated, fitness is modified here.
    def run_single(self, levels, logger, render=False, agent_x=None, agent_y=None, index=0) -> Tuple[float, np.array]:
        loading_progress = ['■ □ □','□ ■ □','□ □ ■']
        levels_to_run = len(levels)
        self.fitness = 0
        for selected_level in range(levels_to_run):
            env = MarioEnvironment.create_mario_environment(levels[selected_level])
            global next_state
            state, old_info = env.reset()
            agent_parameters = self.nn.agent_parameters
            n_episodes = agent_parameters['n_episodes']
            self.nn.to(self.nn.device)
            for episode in range(n_episodes):
                if logger is not None:
                    logger.print(f'Calculating Individual {index} Fitness - {round(self.fitness, ndigits=2)}')
                    logger.tick()
                else:
                    update = int(time.time()) % len(loading_progress)
                    sys.stdout.write("\r | Calculating Fitness " + loading_progress[update] + " | ")
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
                reward = min(agent_parameters['reward_max_x_change'], max(-agent_parameters['reward_max_x_change'],
                                                                          info['x_pos'] - old_info[
                                                                              'x_pos']))  # Clip the x_pos difference to deal with warp points, etc.
                old_info = info
                reward /= 100  # 15
                self.fitness += reward

                # Format the generic agent data to ensure it's compatible with Reinforcement Agents' memory
                reward = torch.tensor([reward])
                action = torch.tensor([[action]], device=self.nn.device, dtype=torch.long)

                if self.replay_memory is not None:
                    self.replay_memory.push(state, action, next_state, reward, not done, not info['flag_get'])

                if isinstance(agent_x, list):
                    agent_x.append(episode * selected_level)
                    agent_y.append(self.fitness / levels_to_run)

                state = next_state
                if done:
                    break

            env.close()

            if logger is not None:
                self.estimate = str(logger.get_estimate())

        self.nn.to(torch.device('cpu'))

        return self.fitness, self.nn.get_weights_biases()


# Convolutional Neural Network Individual
class ReinforcementCNNIndividual(Individual):
    def __init__(self, parameters, memory):
        super().__init__(parameters)
        if self.get('q_val_plot_freq') > 0:
            self.q_values_plot = Dialog()
        self.fitness_plot = Dialog()
        self.temp = True
        self.fitness = 0.0
        self.estimate = 'NA'
        self.replay_memory = memory
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

        # # Generates the first image of all states in the state buffer
        # if self.temp:
        #     preproc.generate_image(84, 84, state_batch[0][0], 0)
        #     preproc.generate_image(84, 84, state_batch[0][1], 1)
        #     preproc.generate_image(84, 84, state_batch[0][2], 2)
        #     preproc.generate_image(84, 84, state_batch[0][3], 3)
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

    def run_single(self, levels, logger, render=False, p=0, agent_x=None, agent_y=None) -> Tuple[float, np.array]:
        global next_state
        self.fitness = 0.0
        old_fitness = 0.0
        steps_done = 0
        moving_fitness = 0
        moving_fitness_mom = 0.95
        moving_fitness_updates = 0
        n_episodes = self.get('n_episodes')
        xp_episodes = self.get('experience_episodes')
        for level in levels:
            env = MarioEnvironment.create_mario_environment(level)
            state, info = env.reset()
            self.nn.to(self.nn.device)
            self.target_nn.to(self.nn.device)
            # Average the fitness result between '
            for episode in range(xp_episodes):  # 'Episodes'
                state, old_info = env.reset()
                old_fitness = self.fitness if old_fitness < self.fitness else old_fitness
                self.fitness = 0.0
                # logger.print(str(episode / xp_episodes) + ' Agent: (R) | Approx: ' + self.estimate)

                for t in range(n_episodes):  # rename to n_steps
                    # logger.tick()
                    # logger.print(str(episode / xp_episodes) + ' Agent: (R) | Approx: ' + self.estimate)
                    if render:
                        env.render()

                    action, steps_done = self.select_action(env, state, steps_done)
                    next_state, reward, done, info = env.step(action)
                    reward = min(self.get('reward_max_x_change'), max(-self.get('reward_max_x_change'),
                                                                      info['x_pos'] - old_info[
                                                                          'x_pos']))  # Clip the x_pos difference to deal with warp points, etc.
                    old_info = info
                    reward /= 100  # 15

                    self.fitness += reward
                    reward = torch.tensor([reward])

                    self.replay_memory.push(state, action, next_state, reward, not done, not info['flag_get'])

                    state = next_state

                    if steps_done >= self.get('learn_start') and steps_done % self.get('train_freq') == 0:
                        self.train_step()

                    if steps_done % self.nn.agent_parameters['target_update_freq'] == 0:
                        self.target_nn.load_state_dict(self.nn.state_dict())

                    if steps_done % self.get('save_freq') == 0:
                        if self.get('q_val_plot_freq') > 0:
                            self.q_values_plot.save_image(self.get('log_dir') + '/graphs/')
                        self.fitness_plot.save_image(self.get('log_dir') + '/graphs/')
                        output_filename = self.get('log_dir') + '/models/RL_agent_' + datetime.now().strftime(
                            "%m%d%Y_%H_%M_%S") + '_' + str(steps_done) + '.npy'
                        np.save(output_filename, self.nn.get_weights_biases())

                    if done:
                        break

                print("Steps: " + str(steps_done) + ", fitness: " + str(self.fitness) + ', got flag: ' + str(info['flag_get']))

                # Plot smoothed running average of fitness
                moving_fitness = moving_fitness_mom * moving_fitness + (1.0 - moving_fitness_mom) * self.fitness
                moving_fitness_updates += 1
                zero_debiased_moving_fitness = moving_fitness / (1.0 - moving_fitness_mom ** moving_fitness_updates)
                self.fitness_plot.add_data_point("fitness", steps_done, [zero_debiased_moving_fitness], False, True)
                self.fitness_plot.update_image('')

                if agent_x is not None:
                    # logger.print_progress(episode)
                    agent_x.append(episode)
                    agent_y.append(self.fitness)

                # Reset the estimate after we've finished one training cycle
            env.close()
        # Help clear GPU memory, by moving network back to the cpu once training run is complete
        self.nn.to(torch.device('cpu'))
        self.target_nn.to(torch.device('cpu'))

        return self.fitness, self.nn.get_weights_biases()
