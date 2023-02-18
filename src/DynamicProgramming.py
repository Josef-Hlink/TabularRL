#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float
        ) -> None:
        ''' Initialize the Q-value iteration agent '''
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        
    def select_action(self, s: int) -> int:
        ''' Returns the greedy best action in state s ''' 
        return argmax(self.Q_sa[s])
    
    def update(
        self,
        s: int,
        a: int,
        p_sas: np.ndarray,
        r_sas: np.ndarray
        ) -> None:
        ''' Function updates Q(s, a) using p_sas and r_sas '''
        Q_sa_copy = self.Q_sa.copy()
        self.Q_sa[s,a] = 0
        for s_ in range(self.n_states):
            self.Q_sa[s,a] += p_sas[s,a,s_] * (r_sas[s,a,s_] + self.gamma * np.max(Q_sa_copy[s_]))
        return None
    
def Q_value_iteration(
    env: StochasticWindyGridworld,
    gamma: float = 1.0,
    threshold: float = 0.001
    ) -> QValueIterationAgent:
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QVIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
    
    i = 0
    while True:
        delta = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                x = QVIagent.Q_sa[s, a]
                QVIagent.update(s, a, env.p_sas, env.r_sas)
                delta = max(delta, abs(x - QVIagent.Q_sa[s, a]))
        if delta < threshold or i > 1000:
            print('Converged')
            break
        env.render(Q_sa = QVIagent.Q_sa, plot_optimal_policy = True, step_pause = 0.2)
        print(f'i = {i}, Δ = {delta:.3f}')
        i += 1
     
    return QVIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model = True)
    env.render()
    QVIagent = Q_value_iteration(env, gamma, threshold)
    
    # View optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QVIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa = QVIagent.Q_sa, plot_optimal_policy = True, step_pause = 0.5)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    experiment()
