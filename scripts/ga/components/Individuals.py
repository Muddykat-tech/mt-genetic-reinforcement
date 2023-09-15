import copy
import math
import random
from abc import ABC, abstractmethod
from collections import namedtuple, deque
from itertools import count
from typing import Tuple

import numpy as np
import torch
from torch import optim

from nn.agents.CNN import CNN
from nn.agents.NeuralNetwork import NeuralNetwork
from nn.preprocess import preproc


# Abstract Individual Class that has base functions the GA can call upon
class Individual(ABC):
    def __init__(self, parameters):
        self.nn = self.get_model(parameters)
        self.fitness = 0.0
        self.weights_biases: np.array = None

    def calculate_fitness(self, env, logger, render, p) -> None:
        self.fitness, self.weights_biases = self.run_single(env, logger, render, p)

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)

    @abstractmethod
    def get_model(self, parameters) -> CNN:
        pass

    @abstractmethod
    def run_single(self, env, logger, render=False, p=0) -> Tuple[float, np.array]:
        pass


# Convolutional Neural Network Individual
class CNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.estimate = 'NA'

    def get_model(self, parameters) -> CNN:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    # This is where actions the agent take are calculated, fitness is modified here.
    def run_single(self, env, logger, render=False, p=0) -> Tuple[float, np.array]:
        done = False
        state = env.reset()
        fitness = 0
        agent_parameters = self.nn.agent_parameters
        n_episodes = agent_parameters['n_episodes']
        self.nn.to(self.nn.device)
        for episode in range(n_episodes):
            logger.tick()
            logger.print('Generic Agent ' + str(p) + ' - Fitness: (' + str(round(fitness)) + ') - Approx Training '
                                                                                             'Time: ' + self.estimate)
            if render:
                env.render()

            state = state.to(self.nn.device)

            # Determine the action
            action_probability = torch.nn.functional.softmax(self.nn.forward(state).mul(agent_parameters['action_conf']),
                                                             dim=1)
            m = torch.distributions.Categorical(action_probability)
            action = m.sample().item()

            # Repeat the action for a few frames
            for _ in range(agent_parameters['n_repeat']):
                state, reward, done, _ = env.step(action)
                reward /= 15
                fitness += reward
                if done:
                    break

            if done:
                break

        self.estimate = str(logger.get_estimate())
        self.nn.to(torch.device('cpu'))

        return fitness, self.nn.get_weights_biases()

# Convolutional Neural Network Individual
class ReinforcementCNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.temp = True
        self.fitness = 0.0
        self.estimate = 'NA'
        self.replay_memory = ReplayMemory(parameters['memory_size'])
        self.target_nn = self.get_model(parameters)
        self.optimizer = optim.AdamW(self.nn.parameters(), lr=1e-4, amsgrad=True)
        self.frames = np.zeros(
            shape=(1, parameters['n_frames'], parameters['downsample_w'], parameters['downsample_h']))

    def get_model(self, parameters) -> CNN:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    def select_action(self, env, state, steps_done):
        sample = random.random()
        eps_threshold = self.get('ep_end') + (self.get('ep_start') - self.get('ep_end')) * \
                        math.exp(-1. * steps_done / self.get('ep_decay'))
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                processed_state = state.to(self.nn.device)
                return self.nn(processed_state).max(1)[1].view(1, 1), steps_done
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.nn.device, dtype=torch.long), steps_done

    def train_step(self):
        batch_size = self.get('batch_size')
        if len(self.replay_memory) < batch_size:
            return

        transitions = self.replay_memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.nn.device,
                                      dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state).to(self.nn.device)
        action_batch = torch.cat(batch.action).to(self.nn.device)
        reward_batch = torch.cat(batch.reward).to(self.nn.device)

        # Generates the first image of all states in the state buffer
        if self.temp:
            preproc.generate_image(84, 84, state_batch[0][0], 0)
            preproc.generate_image(84, 84, state_batch[0][1], 1)
            preproc.generate_image(84, 84, state_batch[0][2], 2)
            preproc.generate_image(84, 84, state_batch[0][3], 3)
            self.temp = False

        state_action_values = self.nn(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.nn.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_nn(non_final_next_states.to(self.nn.device)).max(1)[0]

        expected_state_action_values = (next_state_values * self.get('gamma') + reward_batch)

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.nn.parameters(), 100)
        self.optimizer.step()

    def run_single(self, env, logger, render=False, agent_x=None, agent_y=None) -> Tuple[float, np.array]:
        global next_state
        self.fitness = 0.0
        old_fitness = 0.0
        steps_done = 0
        n_episodes = self.get('n_episodes')
        xp_episodes = self.get('experience_episodes')

        self.nn.to(self.nn.device)
        self.target_nn.to(self.nn.device)
        # Average the fitness result between '
        for episode in range(xp_episodes):  # 'Episodes'
            state = env.reset()
            old_fitness = self.fitness if old_fitness < self.fitness else old_fitness
            self.fitness = 0.0
            logger.print(str(episode / xp_episodes) + ' Agent: (R) | Approx: ' + self.estimate)

            for t in range(n_episodes):  # rename to n_steps
                logger.tick()
                logger.print(str(episode / xp_episodes) + ' Agent: (R) | Approx: ' + self.estimate)
                if render:
                    env.render()

                action, steps_done = self.select_action(env, state, steps_done)
                reward = 0
                for _ in range(self.get('n_repeat')):
                    next_state, tmp_reward, done, info = env.step(action.item())
                    tmp_reward /= 15
                    reward += tmp_reward

                    if done:
                        break

                self.fitness += reward
                reward = torch.tensor([reward])

                self.replay_memory.push(state, action, next_state, reward)

                state = next_state

                if steps_done >= self.get('learn_start'):
                    self.train_step()

                if t % self.nn.agent_parameters['target_update_freq'] == 0:
                    self.target_nn.load_state_dict(self.nn.state_dict())

                    # target_net_state_dict = self.target_nn.state_dict()
                    # policy_net_state_dict = self.nn.state_dict()
                    # for key in policy_net_state_dict:
                    #     target_net_state_dict[key] = policy_net_state_dict[key] * self.get('tau') + \
                    #                                  target_net_state_dict[
                    #                                      key] * (1 - self.get('tau'))
                    #     self.target_nn.load_state_dict(target_net_state_dict)

                if done:
                    break

            if agent_x is not None:
                logger.print_progress(episode)
                agent_x.append(episode)
                agent_y.append(self.fitness)

            # Reset the estimate after we've finished one training cycle
            self.estimate = str(logger.get_estimate())

        # Help clear GPU memory, by moving network back to the cpu once training run is complete
        self.nn.to(torch.device('cpu'))
        self.target_nn.to(torch.device('cpu'))

        return self.fitness, self.nn.get_weights_biases()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
