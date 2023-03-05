#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
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
        for _ in range(100):
            a = argmax(self.Q[s])
            s, r, done = env.step(a)
            rewards.append(r)
            if done:
                break
        return np.mean(rewards)

    def plot_policy(self, title: str, filename: str) -> None:
        ''' Plot the policy based on the current state of self.Q
        Each state is represented by an arrow pointing in the direction of the action with the highest value
        The state is plotted as a 2D representation of the environment with a BW heatmap of the greedy Q-values
        '''
        Q_2D = self.Q.reshape(10, 7, self.n_actions)
        Q_2D = np.transpose(Q_2D, (1, 0, 2))
        Q_2D = np.flip(Q_2D, axis=0)
        # get the greedy action for each state (can be multiple)
        greedy_actions = np.argmax(Q_2D, axis=2, keepdims=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        # plot the heatmap of the greedy Q-values
        ax.imshow(np.max(Q_2D, axis=2), cmap='gray')
        
        # plot the arrows
        for i in range(7):
            for j in range(10):
                # get the greedy action
                a = greedy_actions[i, j, 0]
                # get the direction of the arrow
                if a == 0:
                    dx, dy = 0, -0.2
                elif a == 1:
                    dx, dy = 0.2, 0
                elif a == 2:
                    dx, dy = 0, 0.2
                else:
                    dx, dy = -0.2, 0
                # plot the arrow
                ax.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')

        # set markings where wind is blowing (red arrow)
        # columns 3-5 with strength 1, columns 6-7 with strength 2 and column 8 with strength 1
        ax.plot([2.5, 5.5], [-.5, -.5], 'r', lw=3)
        ax.plot([2.5, 5.5], [6.5, 6.5], 'r', lw=3)
        ax.plot([2.5, 2.5], [-.5, 6.5], 'r', lw=3)
        ax.plot([5.5, 5.5], [-.5, 6.5], 'r', lw=3)
        
        ax.plot([5.5, 7.5], [-.5, -.5], 'r', lw=5)
        ax.plot([5.5, 7.5], [6.5, 6.5], 'r', lw=5)
        ax.plot([7.5, 7.5], [-.5, 6.5], 'r', lw=5)
        ax.plot([5.5, 5.5], [-.5, 6.5], 'r', lw=5)

        ax.plot([7.5, 8.5], [-.5, -.5], 'r', lw=3)
        ax.plot([7.5, 8.5], [6.5, 6.5], 'r', lw=3)
        ax.plot([7.5, 7.5], [-.5, 6.5], 'r', lw=3)
        ax.plot([8.5, 8.5], [-.5, 6.5], 'r', lw=3)

        # set markings where the goal is (green circle)
        ax.plot([7], [3], 'go', markersize=50)

        ax.set_title(title, fontsize=16)
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
