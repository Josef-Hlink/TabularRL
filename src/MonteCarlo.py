#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Optional
import numpy as np
from Environment import StochasticWindyGridworld
from Agent import Agent


class MonteCarloAgent(Agent):
        
    def update(
        self,
        states: list[int],
        actions: list[int],
        rewards: list[float],
        ) -> None:
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        assert len(states) == len(actions) + 1 and len(actions) == len(rewards)
        states.pop()
        G = 0
        for s, a, r in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = self.gamma * G + r
            self.Q[s,a] += self.learning_rate * (G - self.Q[s,a])
        return

def monte_carlo(
    n_timesteps: int,
    max_episode_length: int,
    learning_rate: float,
    gamma: float,
    policy: str = 'egreedy',
    epsilon: Optional[float] = None,
    temp: Optional[float] = None
    ) -> np.ndarray:
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = np.zeros(n_timesteps)
    greedy_rewards = np.zeros(n_timesteps // 500)

    for t in range(n_timesteps):
        s = env.reset()
        states, actions, rewards_ep = [s], [], []
        # playout
        for _ in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            actions.append(a)
            s, r, done = env.step(a)
            states.append(s)
            rewards_ep.append(r)
            if done:
                break
        pi.update(states, actions, rewards_ep)
        rewards[t] = np.sum(rewards_ep)
        if t % 500 == 0:
            dummy_env = StochasticWindyGridworld(initialize_model=False)
            greedy_rewards[t//500] = pi.run_greedy_episode(dummy_env)

    return (rewards, greedy_rewards)

def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    start = time.time()

    rewards, greedy_rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, policy, epsilon, temp)
    print(f'Average reward: {np.mean(rewards)}')
    print(f'Last greedy reward: {greedy_rewards[-1]}')
    print(f'Time taken: {time.time() - start:.2f} seconds')



if __name__ == '__main__':
    test()
