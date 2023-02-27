#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class Agent:
    ''' Base class for agents '''

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float,
        gamma: float
        ) -> None:
        ''' Initialize the agent '''
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))

    def select_action(
        self,
        s: int,
        policy: str = 'egreedy',
        epsilon: Optional[float] = None,
        temp: Optional[float] = None
        ) -> int:

        if policy == 'egreedy':
            if epsilon is None:
                raise ValueError('Provide an epsilon to use egreedy policy')
            a = self.select_egreedy_action(s, epsilon)
        elif policy == 'softmax':
            if temp is None:
                raise ValueError('Provide a temperature to use softmax policy')
            a = self.select_softmax_action(s, temp)
        else:
            raise ValueError('Unknown policy, please use egreedy or softmax')

        return a

    def select_egreedy_action(
        self,
        s: int,
        epsilon: float
        ) -> int:
        ''' Returns an action according to the epsilon-greedy policy '''
        greedy_a = argmax(self.Q[s])
        if np.random.uniform() < epsilon:
            explore_a = np.random.randint(0, self.n_actions)
            while explore_a == greedy_a:
                explore_a = np.random.randint(0, self.n_actions)
            return explore_a
        return greedy_a

    def select_softmax_action(
        self,
        s: int,
        temp: float
        ) -> int:
        ''' Returns an action according to the softmax policy '''
        return np.random.choice(self.n_actions, p=softmax(self.Q[s], temp))

    def update(self) -> None:
        raise NotImplementedError
    
    def run_greedy_episode(self, env: StochasticWindyGridworld) -> float:
        ''' Run a greedy episode and returns the average reward per timestep '''
        s = env.reset()
        done = False
        rewards = []
        while not done:
            a = self.select_action(s, policy='egreedy', epsilon=0)
            s, r, done = env.step(a)
            rewards.append(r)
        return np.mean(rewards)
