#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
from Agent import Agent

class QValueIterationAgent(Agent):
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''
        
    def select_action(self, s: int) -> int:
        ''' Returns the greedy best action in state s ''' 
        return argmax(self.Q[s])
    
    def update(
        self,
        s: int,
        a: int,
        p_sas: np.ndarray,
        r_sas: np.ndarray
        ) -> None:
        ''' Update Q(s, a) using p_sas and r_sas '''
        Q_copy = self.Q.copy()
        self.Q[s,a] = 0
        for s_ in range(self.n_states):
            self.Q[s,a] += p_sas[s,a,s_] * (r_sas[s,a,s_] + self.gamma * np.max(Q_copy[s_]))
        return None
    
def Q_value_iteration(
    env: StochasticWindyGridworld,
    gamma: float = 1.0,
    threshold: float = 0.001,
    verbose: bool = False,
    plot: bool = False
    ) -> QValueIterationAgent:
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QVIagent = QValueIterationAgent(env.n_states, env.n_actions, learning_rate=None, gamma=gamma)
    
    i = 0
    while True:
        delta = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                x = QVIagent.Q[s, a]
                QVIagent.update(s, a, env.p_sas, env.r_sas)
                delta = max(delta, abs(x - QVIagent.Q[s, a]))
        if plot:
            QVIagent.plot_policy(f'DP iteration {i}', f'../plots/DP/DP{i}.png')
        if verbose:
            print(f'i = {i}, Δ = {delta:.3f}')
        if delta < threshold or i > 1000:
            break
        i += 1
    
     
    return QVIagent

def experiment(verbose: bool = False, plot: bool = False) -> float:
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model = True)
    QVIagent = Q_value_iteration(env, gamma, threshold, verbose, plot)
    rewards = []
    # env.render()
    
    # View optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QVIagent.select_action(s)
        s_next, r, done = env.step(a)
        # env.render(Q=QVIagent.Q,plot_optimal_policy=True,step_pause=3)
        rewards.append(r)
        s = s_next

    if verbose:
        print(f'Average reward: {np.mean(rewards):.3f}')

    return np.mean(rewards)

if __name__ == '__main__':
    experiment(verbose=True, plot=True)
