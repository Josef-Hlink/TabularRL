#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
from typing import Optional
import numpy as np
import time

from DynamicProgramming import experiment as dp_experiment
from Q_learning import q_learning
from SARSA import sarsa
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(
    backup: str,
    n_repetitions: int,
    n_timesteps: int,
    max_episode_length: int,
    learning_rate: float,
    gamma: float,
    policy: str = 'egreedy',
    epsilon: Optional[float] = None,
    temp: Optional[float] = None,
    smoothing_window: int = 51,
    plot: bool = False,
    n: int = 5
    ) -> np.ndarray:
    ''' Runs the specified algorithm multiple times and returns the average learning curve '''
    onestep_args = [n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot]
    nstep_args = [n_timesteps, max_episode_length, learning_rate, gamma, policy, epsilon, temp, plot]
    algorithms = {
        'q': (q_learning, onestep_args),
        'sarsa': (sarsa, onestep_args),
        'mc': (monte_carlo, nstep_args),
        'nstep': (n_step_Q, nstep_args + [n])
    }
    reward_results = np.empty([n_repetitions, n_timesteps])
    
    start = time.time()
    for rep in range(n_repetitions):

        algorithm, args = algorithms[backup]
        rewards = algorithm(*args)
        reward_results[rep] = rewards
    
    print(f'backup: {backup}, policy: {policy}, epsilon: {epsilon}, temp: {temp}, lr: {learning_rate}')
    print(f'time: {time.strftime("%M:%S", time.gmtime(time.time() - start))}')
    learning_curve = np.mean(reward_results, axis = 0)
    learning_curve = smooth(learning_curve, smoothing_window)
    return learning_curve  

def experiment():
    ####### Settings

    backup_labels = {
        'q': 'Q-learning',
        'sarsa': 'SARSA',
        'mc': 'Monte Carlo',
        'nstep': 'n-step Q-learning'
    }

    run_repetitions = partial(
        average_over_repetitions,
        n_repetitions = 5, #10
        n_timesteps = 50000, #50000
        max_episode_length = 150,
        gamma = 1.0,
        smoothing_window = 1001,
        plot = False,
        learning_rate = 0.25,
        policy = 'egreedy',
        epsilon = 0.1,
        temp = None,
        n = 5
    )

    ####### Experiments
    
    ###### Assignment 1: Dynamic Programming
    r_avg_opt = dp_experiment(verbose = False)

    ###### Assignment 2: Effect of exploration
    Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration')    
    for epsilon in [0.02, 0.1, 0.3]:
        Plot.add_curve(
            y = run_repetitions(
                backup = 'q',
                learning_rate = 0.25,
                policy = 'egreedy',
                epsilon = epsilon,
                temp = None
            ),
            label = rf'$\epsilon$-greedy, $\epsilon $ = {epsilon}'
        )
    for temp in [0.01, 0.1, 1.0]:
        Plot.add_curve(
            y = run_repetitions(
                backup = 'q',
                learning_rate = 0.25,
                policy = 'softmax',
                epsilon = None,
                temp = temp
            ),
            label = rf'softmax, $ \tau $ = {temp}'
            )
    Plot.add_hline(r_avg_opt, label = 'DP optimum')
    Plot.save('exploration.png')
    
    ###### Assignment 3: Q-learning versus SARSA
    Plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')    
    for backup in ['q', 'sarsa']:
        for learning_rate in [0.02, 0.1, 0.4]:
            Plot.add_curve(
                y = run_repetitions(
                    backup = backup,
                    learning_rate = learning_rate,
                    policy = 'egreedy',
                    epsilon = 0.1,
                    temp = None
                ),
                label = rf'{backup_labels[backup]}, $\alpha$ = {learning_rate}'
            )
    Plot.add_hline(r_avg_opt, label = 'DP optimum')
    Plot.save('on_off_policy.png')
    
    ###### Assignment 4: Back-up depth
    Plot = LearningCurvePlot(title = 'Back-up: depth')    
    for n in [1, 3, 10, 30]:
        Plot.add_curve(
            y = run_repetitions(backup = 'nstep', n = n),
            label = rf'{n}-step Q-learning'
        )
    backup = 'mc'
    Plot.add_curve(
        y = run_repetitions(backup = 'mc'),
        label = 'Monte Carlo'
    )
    Plot.add_hline(r_avg_opt, label = 'DP optimum')
    Plot.save('depth.png')


if __name__ == '__main__':
    experiment()
