#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class QLearningAgent:

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 1.0
        ) -> None:
        ''' Initialize the Q-learning agent '''
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        
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
        greedy_a = argmax(self.Q_sa[s])
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
        return np.random.choice(self.n_actions, p=softmax(self.Q_sa[s], temp))

    def update(self, s : int, a: int, r: int, s_: int, done: bool = False) -> None:
        ''' Tabular Q-learning update '''
        G = r + self.gamma * np.max(self.Q_sa[s_]) * (1 - done)
        self.Q_sa[s,a] += self.learning_rate * (G - self.Q_sa[s,a])
        return None

def q_learning(
    n_timesteps: int,
    learning_rate: float,
    gamma: float,
    policy: str = 'egreedy',
    epsilon: Optional[float] = None,
    temp: Optional[float] = None,
    plot: bool = True
    ) -> np.ndarray:
    ''' runs a single repetition of Q-learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model = False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = np.zeros(n_timesteps)

    s = env.reset()
    for t in range(n_timesteps):
        a = pi.select_action(s, policy, epsilon, temp)
        s_, r, done = env.step(a)
        pi.update(s, a, r, s_, done)
        rewards[t] = r
        s = s_
        if done:
            s = env.reset()
    
        if plot:
            # Plot the Q-value estimates during Q-learning execution
            env.render(pi.Q_sa, plot_optimal_policy = True, step_pause = 0.1)

    return rewards 

def test():
    
    n_timesteps = 10000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False
    # plot = True

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print('Number of times reached goal:', np.sum(rewards == 40))


if __name__ == '__main__':
    test()
