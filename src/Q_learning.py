#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from Environment import StochasticWindyGridworld
from Agent import Agent


class QLearningAgent(Agent):

    def update(self, s : int, a: int, r: int, s_: int, done: bool = False) -> None:
        ''' Tabular Q-learning update '''
        G = r + self.gamma * np.max(self.Q[s_]) * (1 - done)
        self.Q[s,a] += self.learning_rate * (G - self.Q[s,a])
        return None

def q_learning(
    n_timesteps: int,
    learning_rate: float,
    gamma: float,
    policy: str = 'egreedy',
    epsilon: Optional[float] = None,
    temp: Optional[float] = None
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
