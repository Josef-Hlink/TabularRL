#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
from typing import Optional
import numpy as np
import time
from multiprocessing import Pool

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
    n: int = 5
    ) -> np.ndarray:
    ''' Runs the specified algorithm multiple times and returns the average learning curve '''
    args_1step = [n_timesteps, learning_rate, gamma, policy, epsilon, temp]
    args_nstep = [n_timesteps, max_episode_length, learning_rate, gamma, policy, epsilon, temp]
    algorithms = {
        'q': (q_learning, args_1step),
        'sarsa': (sarsa, args_1step),
        'mc': (monte_carlo, args_nstep),
        'nstep': (n_step_Q, args_nstep + [n])
    }
    reward_results = np.zeros((n_repetitions, n_timesteps))
    
    start = time.time()

    algorithm, args = algorithms[backup]
    with Pool() as p:
        for rep, rewards in enumerate(p.starmap(algorithm, [(*args,) for _ in range(n_repetitions)])):
            reward_results[rep] = rewards
    
    signature = f'backup: {backup}, policy: {policy}, epsilon: {epsilon}, temp: {temp}, lr: {learning_rate}'
    signature = signature + f', n: {n}' if backup == 'nstep' else signature
    print(f'{signature}\ntime: {time.strftime("%M:%S", time.gmtime(time.time() - start))}')
    
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
        n_repetitions = 50,
        n_timesteps = 50000, #50000
        max_episode_length = 150,
        gamma = 1.0,
        smoothing_window = 1001,
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
    plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration')    
    for epsilon in [0.02, 0.1, 0.3]:
        plot.add_curve(
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
        plot.add_curve(
            y = run_repetitions(
                backup = 'q',
                learning_rate = 0.25,
                policy = 'softmax',
                epsilon = None,
                temp = temp
            ),
            label = rf'softmax, $ \tau $ = {temp}'
        )
    plot.add_hline(r_avg_opt, label = 'DP optimum')
    plot.save('exploration.png')
    
    ###### Assignment 3: Q-learning versus SARSA
    plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')    
    for backup in ['q', 'sarsa']:
        for learning_rate in [0.02, 0.1, 0.4]:
            plot.add_curve(
                y = run_repetitions(
                    backup = backup,
                    learning_rate = learning_rate,
                    policy = 'egreedy',
                    epsilon = 0.1,
                    temp = None
                ),
                label = rf'{backup_labels[backup]}, $\alpha$ = {learning_rate}'
            )
    plot.add_hline(r_avg_opt, label = 'DP optimum')
    plot.save('on_off_policy.png')
    
    ###### Assignment 4: Back-up depth
    plot = LearningCurvePlot(title = 'Back-up: depth')    
    for n in [1, 3, 10, 30]:
        plot.add_curve(
            y = run_repetitions(backup = 'nstep', n = n),
            label = rf'{n}-step Q-learning'
        )
    plot.add_curve(
        y = run_repetitions(backup = 'mc'),
        label = 'Monte Carlo'
    )
    plot.add_hline(r_avg_opt, label = 'DP optimum')
    plot.save('depth.png')


if __name__ == '__main__':
    experiment()
