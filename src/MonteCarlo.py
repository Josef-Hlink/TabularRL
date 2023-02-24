#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
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
        
    def update(
        self,
        states: list,
        actions: list,
        rewards: list,
        ) -> None:
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        assert len(states) == len(actions) + 1 and len(actions) == len(rewards)
        G = 0
        for t in range(len(rewards)-1, -1, -1):
            G = self.gamma * G + rewards[t]
            self.Q_sa[states[t]][actions[t]] += self.learning_rate * (G - self.Q_sa[states[t]][actions[t]])
        return

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    for t in range(n_timesteps):
        s = env.reset()
        states, actions, rewards_ep = [s], [], []
        for t_ep in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            actions.append(a)
            s, r, done = env.step(a)
            states.append(s)
            rewards_ep.append(r)
            if done:
                break
        pi.update(states, actions, rewards_ep)
        rewards.append(sum(rewards_ep))

    return rewards 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))  
            
if __name__ == '__main__':
    test()
