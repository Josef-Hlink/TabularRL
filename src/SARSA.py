#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from Environment import StochasticWindyGridworld
from Agent import Agent


class SarsaAgent(Agent):

    def update(self, s : int, a: int, r: int, s_: int, a_: int, done: bool = False) -> None:
        ''' Tabular SARSA update '''
        G = r + self.gamma * self.Q[s_,a_] * (1 - done)
        self.Q[s,a] += self.learning_rate * (G - self.Q[s,a])
        return None


def sarsa(
    n_timesteps: int,
    learning_rate: float,
    gamma: float,
    policy: str = 'egreedy',
    epsilon: Optional[float] = None,
    temp: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray]:
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 

    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = np.zeros(n_timesteps)
    greedy_rewards = np.zeros(n_timesteps // 500)

    s = env.reset()
    a = pi.select_action(s, policy, epsilon, temp)
    for t in range(n_timesteps):
        s_, r, done = env.step(a)
        a_ = pi.select_action(s_, policy, epsilon, temp)
        pi.update(s, a, r, s_, a_, done)
        s, a = s_, a_
        rewards[t] = r
        if done:
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
        if t % 500 == 0:
            dummy_env = StochasticWindyGridworld(initialize_model=False)
            greedy_rewards[t//500] = pi.run_greedy_episode(dummy_env)

    return (rewards, greedy_rewards)


def test():
    n_timesteps = 10000
    gamma = 1.0
    learning_rate = 0.1
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0

    rewards, greedy_rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp)
    print(f'Number of times reached goal: {np.sum(rewards == 40)}')
    print(f'Last greedy reward: {greedy_rewards[-1]}')


if __name__ == '__main__':
    test()
