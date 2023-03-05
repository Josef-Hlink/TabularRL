#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import Agent

import time


class NstepQLearningAgent(Agent):

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float,
        gamma: float,
        n: int
        ) -> None:
        super().__init__(n_states, n_actions, learning_rate, gamma)
        self.n = n
        self.discount_array = self.gamma ** np.arange(0, self.n)
        
    def update(
        self,
        states: list[int],
        actions: list[int],
        rewards: list[float],
        done: bool
        ) -> None:
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(states) - 1
        rewards = np.array(rewards)
        for t in range(T_ep):
            m = min(self.n, T_ep-t)
            # target is the sum of (potentially discounted) rewards from t+1 to t+m
            G = np.sum(self.discount_array[:m] * rewards[t:t+m])
            if not (done and t+m == T_ep):
                # target should also include the Q-learning estimate of the value of the state at time t+m
                G += self.gamma**m * np.max(self.Q[states[t+m]])
            self.Q[states[t], actions[t]] += self.learning_rate * (G - self.Q[states[t], actions[t]])
        return

def n_step_Q(
    n_timesteps: int,
    max_episode_length: int,
    learning_rate: float,
    gamma: float,
    policy: str = 'egreedy',
    epsilon: float = None,
    temp: float = None,
    n: int = 5,
    plot: bool = False
    ) -> np.ndarray:
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    rewards = np.zeros(n_timesteps)
    greedy_rewards = np.zeros(n_timesteps // 500)

    for t in range(n_timesteps):
        s = env.reset()
        states, actions, rewards_ep = [s], [], []
        # playout
        for _ in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s, r, done = env.step(a)
            states.append(s)
            actions.append(a)
            rewards_ep.append(r)
            if done:
                break
        pi.update(states, actions, rewards_ep, done)
        rewards[t] = np.mean(rewards_ep)

        if t % 500 == 0:
            dummy_env = StochasticWindyGridworld(initialize_model=False)
            greedy_rewards[t//500] = pi.run_greedy_episode(dummy_env)
    
    if plot:
        pi.plot_policy(f'{n}-step after {n_timesteps} episodes', f'../plots/NS/NS{n}.png')
    return (rewards, greedy_rewards)

def test():
    n_timesteps = 5000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 30
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    plot = True
    
    start = time.time()
    rewards, greedy_rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, policy, epsilon, temp, n, plot)
    
    print(f'Average reward: {np.mean(rewards)}')
    print(f'Last greedy reward: {greedy_rewards[-1]}')
    print(f'Time elapsed: {time.time() - start:.2f} seconds')


if __name__ == '__main__':
    test()
