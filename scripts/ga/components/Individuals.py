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


# Abstract Individual Class that has base functions the GA can call upon
class Individual(ABC):
    def __init__(self, parameters):
        self.nn = self.get_model(parameters)
        self.fitness = 0.0
        self.weights_biases: np.array = None

    def calculate_fitness(self, env, logger, render) -> None:
        self.fitness, self.weights_biases = self.run_single(env, logger, render)

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)

    @abstractmethod
    def get_model(self, parameters) -> CNN:
        pass

    @abstractmethod
    def run_single(self, env, logger, render=False) -> Tuple[float, np.array]:
        pass


# Convolutional Neural Network Individual
class CNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)

    def get_model(self, parameters) -> CNN:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    # This is where actions the agent take are calculated, fitness is modified here.
    def run_single(self, env, logger, render=False) -> Tuple[float, np.array]:
        done = False
        state = env.reset()
        fitness = 0
        old_fitness = None
        agent_parameters = self.nn.agent_parameters
        n_episodes = agent_parameters['n_episodes']
        frames = np.zeros(
            shape=(1, agent_parameters['n_frames'], agent_parameters['downsample_w'], agent_parameters['downsample_h']))
        deadrun = n_episodes / 2
        checkfit = int(round(n_episodes / 3))
        for episode in range(n_episodes):
            logger.tick()

            if render:
                env.render()
            obs = torch.from_numpy(state.copy()).float()

            # Update frames
            processed_state = self.nn.preprocess.forward(obs[episode % agent_parameters['n_frames'],])
            frames[0, episode % agent_parameters['n_frames']] = processed_state
            data = torch.from_numpy(frames).to(self.nn.device)

            # Determine the action
            action_probability = torch.nn.functional.softmax(self.nn.forward(data).mul(agent_parameters['action_conf']),
                                                             dim=1)
            m = torch.distributions.Categorical(action_probability)
            action = m.sample().item()

            # Repeat the action for a few frames
            for _ in range(agent_parameters['n_repeat']):
                obs, reward, done, _ = env.step(action)
                fitness += reward
                if done:
                    break

            # If the fitness isn't improved much at the start, it's likely it got stuck standing still or something
            # so just kill it then and there, this could be better checked by comparing previous results from like
            # a few episodes ago
            if episode > checkfit:
                old_fitness = fitness

            if episode > deadrun and fitness <= old_fitness:
                break

            if done:
                break

        return fitness, self.nn.get_weights_biases()


# Convolutional Neural Network Individual
class ReinforcementCNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.fitness = 0.0
        self.replay_memory = ReplayMemory(parameters['memory_size'])
        self.target_nn = self.get_model(parameters)
        self.optimizer = optim.AdamW(self.nn.parameters(), lr=1e-4, amsgrad=True)
        self.frames = np.zeros(
            shape=(1, parameters['n_frames'], parameters['downsample_w'], parameters['downsample_h']))

    def get_model(self, parameters) -> CNN:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    def preproc(self, state, episode) -> np.ndarray:
        obs = torch.from_numpy(state.copy()).float()
        processed_state = self.nn.preprocess.forward(obs[episode % self.get('n_frames'), ])
        self.frames[0, episode % self.get('n_frames')] = processed_state
        data = torch.from_numpy(self.frames).to(self.nn.device)
        return data

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
                return self.nn(state).max(1)[1].view(1, 1), steps_done
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

    def run_single(self, env, logger, render=False) -> Tuple[float, np.array]:
        self.fitness = 0.0
        steps_done = 0
        n_episodes = self.get('n_episodes')
        for episode in range(n_episodes):
            logger.tick()
            state = env.reset()
            state = self.preproc(state, episode)
            for t in count():
                action, steps_done = self.select_action(env, state, steps_done)
                observation, reward, terminated, truncated = env.step(action.item())
                self.fitness += reward  # This may need to be changed?
                reward = torch.tensor([reward])
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = self.preproc(observation, t)

                self.replay_memory.push(state, action, next_state, reward)

                state = next_state

                self.train_step()
                target_net_state_dict = self.target_nn.state_dict()
                policy_net_state_dict = self.nn.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.get('tau') + target_net_state_dict[
                        key] * (1 - self.get('tau'))
                    self.target_nn.load_state_dict(target_net_state_dict)

                if done:
                    break

        # Now evaluate the reinforcement agent
        fitness = 0.0
        frames = np.zeros(
            shape=(1, self.get('n_frames'), self.get('downsample_w'), self.get('downsample_h')))
        deadrun = n_episodes / 2
        checkfit = int(round(n_episodes / 3))
        old_fitness = fitness
        state = env.reset()
        for episode in range(n_episodes):
            logger.tick()

            if render:
                env.render()
            obs = torch.from_numpy(state.copy()).float()

            # Update frames
            processed_state = self.nn.preprocess.forward(obs[episode % self.get('n_frames'),])
            frames[0, episode % self.get('n_frames')] = processed_state
            data = torch.from_numpy(frames).to(self.nn.device)

            # Determine the action
            action_probability = torch.nn.functional.softmax(self.nn.forward(data).mul(self.get('action_conf')),
                                                             dim=1)
            m = torch.distributions.Categorical(action_probability)
            action = m.sample().item()

            # Repeat the action for a few frames
            for _ in range(self.get('n_repeat')):
                obs, reward, done, _ = env.step(action)
                fitness += reward
                if done:
                    break

            # If the fitness isn't improved much at the start, it's likely it got stuck standing still or something
            # so just kill it then and there, this could be better checked by comparing previous results from like
            # a few episodes ago
            if episode > checkfit:
                old_fitness = fitness

            if episode > deadrun and fitness <= old_fitness:
                break

            if done:
                break

        return fitness, self.nn.get_weights_biases()


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
