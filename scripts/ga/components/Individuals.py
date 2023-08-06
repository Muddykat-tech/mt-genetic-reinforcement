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
    def get_model(self, parameters) -> NeuralNetwork:
        pass

    @abstractmethod
    def run_single(self, env, logger, render=False) -> Tuple[float, np.array]:
        pass


# Convolutional Neural Network Individual
class CNNIndividual(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)

    def get_model(self, parameters) -> NeuralNetwork:
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
        frames = np.zeros(shape=(1, agent_parameters['n_frames'], agent_parameters['downsample_w'], agent_parameters['downsample_h']))
        deadrun = n_episodes / 2
        checkfit = int(round(n_episodes / 3))
        for episode in range(n_episodes):
            logger.tick()

            if render:
                env.render()
            obs = torch.from_numpy(state.copy()).float()

            # Update frames
            processed_state = self.nn.preprocess.forward(obs[episode % agent_parameters['n_frames'], ])
            frames[0, episode % agent_parameters['n_frames']] = processed_state
            data = torch.from_numpy(frames).to(self.nn.device)

            # Determine the action
            action_probability = torch.nn.functional.softmax(self.nn.forward(data).mul(agent_parameters['action_conf']), dim=1)
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

    def get_model(self, parameters) -> NeuralNetwork:
        return CNN(parameters)

    def get(self, parameter):
        return self.nn.agent_parameters[parameter]

    def optimize_step(self, memory, BATCH_SIZE, fitness, policy_net, target_net, GAMMA, optimizer):
        if len(memory) < BATCH_SIZE:
            return fitness, self.nn.get_weights_biases()

        transitions = memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.nn.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.nn.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    def select_action(self, env, EPS_END, EPS_START, policy_net, state, EPS_DECAY, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                p = policy_net(state).max(1)[1]
                return p.view(1, 1), steps_done
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.nn.device, dtype=torch.long), steps_done

        # This is where actions the agent take are calculated, fitness is modified here.
    def run_single(self, env, logger, render=False) -> Tuple[float, np.array]:
        fitness = 0.0
        # Convert this into special agent params
        BATCH_SIZE = 128
        GAMMA = 0.99
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        TAU = 0.005
        LR = 1e-4
        n_episodes = self.nn.agent_parameters['n_episodes']

        policy_net = copy.deepcopy(self.nn.to(self.nn.device))
        target_net = copy.deepcopy(self.nn.to(self.nn.device))
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)

        steps_done = 0
        agent_parameters = self.nn.agent_parameters
        episode_durations = []

        state = env.reset()
        frames = np.zeros(shape=(1, agent_parameters['n_frames'], agent_parameters['downsample_w'], agent_parameters['downsample_h']))
        for i_episode in range(n_episodes):
            # State Witchcraft that is likely not doing things correctly.
            obs = torch.from_numpy(state.copy()).float()
            processed_state = self.nn.preprocess.forward(obs[i_episode % agent_parameters['n_frames'], ])
            frames[0, i_episode % agent_parameters['n_frames']] = processed_state
            data = torch.from_numpy(frames).to(self.nn.device)

            for t in count():
                action, steps_done = self.select_action(env, EPS_END, EPS_START, policy_net, data, EPS_DECAY, steps_done)
                observation, reward, terminated, truncated = env.step(action.item())
                reward = torch.tensor([reward], device=self.nn.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.nn.device)

                memory.push(torch.from_numpy(state.copy()).to(self.nn.device), action, next_state, reward)
                state = next_state.cpu().numpy()

                self.optimize_step(memory, BATCH_SIZE, fitness, policy_net, target_net, GAMMA, optimizer)
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

                target_net.load_state_dict(target_net_state_dict)

                fitness += reward
                if done:
                    episode_durations.append(t + 1)
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