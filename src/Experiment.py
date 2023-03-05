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
    greedy_reward_results = np.zeros((n_repetitions, n_timesteps // 500))
    
    start = time.time()

    algorithm, args = algorithms[backup]
    with Pool() as p:
        for rep, (rewards, greedy_rewards) in enumerate(p.starmap(algorithm, [(*args,) for _ in range(n_repetitions)])):
            reward_results[rep] = rewards
            greedy_reward_results[rep] = greedy_rewards
    
    signature = f'backup: {backup}, policy: {policy}, epsilon: {epsilon}, temp: {temp}, lr: {learning_rate}'
    signature = signature + f', n: {n}' if backup == 'nstep' else signature
    print(f'{signature}\ntime: {time.strftime("%M:%S", time.gmtime(time.time() - start))}')
    
    learning_curve = np.mean(reward_results, axis = 0)
    if n_timesteps < 50_000:
        sw1, sw2 = 51, 3
    else:
        sw1, sw2 = 501, 31
    learning_curve = smooth(learning_curve, sw1)
    eval_curve = np.mean(greedy_reward_results, axis = 0)
    eval_curve = smooth(eval_curve, sw2)
    return learning_curve, eval_curve

def experiment():
    ####### Settings

    n_repetitions = 50
    n_timesteps = 50000

    run_repetitions = partial(
        average_over_repetitions,
        n_repetitions = n_repetitions,
        n_timesteps = n_timesteps,
        max_episode_length = 150,
        gamma = 1.0,
        smoothing_window = 1001,
        learning_rate = 0.25,
        policy = 'egreedy',
        epsilon = 0.1,
        temp = None,
        n = 5
    )

    greedy_eval_range = np.arange(0, n_timesteps, 500)

    ####### Experiments
    
    ###### Assignment 1: Dynamic Programming
    print('\033[1m' + 'Dynamic Programming' + '\033[0m')
    r_avg_opt = dp_experiment(verbose=False)
    print(f'Average optimal reward: {r_avg_opt:.3f}')

    ###### Assignment 2: Effect of exploration
    print('\033[1m' + 'Exploration' + '\033[0m')
    plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax')
    for i, epsilon in enumerate([0.02, 0.1, 0.3]):
        lc, ec = run_repetitions(backup='q', policy='egreedy', epsilon=epsilon, temp=None)
        plot.add_curve(axid=1, y=lc, cid=i, label=rf'$\epsilon$-greedy, $\epsilon $ = {epsilon}')
        plot.add_curve(axid=2, x=greedy_eval_range, y=ec, cid=i)
    for i, temp in enumerate([0.01, 0.1, 1.0]):
        lc, ec = run_repetitions(backup='q', policy='softmax', epsilon=None, temp=temp)
        plot.add_curve(axid=1, y=lc, cid=i + 3, label=rf'softmax, $ \tau $ = {temp}')
        plot.add_curve(axid=2, x=greedy_eval_range, y=ec, cid=i + 3)
    plot.add_hline(r_avg_opt, label = 'DP optimum')
    plot.set_ylim(-1.1, 1.5)
    plot.set_titles('Learning curve', 'Greedy evaluation')
    plot.save('exploration.png')
    
    ###### Assignment 3: Q-learning versus SARSA
    print('\033[1m' + 'QL vs. SARSA' + '\033[0m')
    plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')    
    for i, backup in enumerate(['q', 'sarsa']):
        for j, learning_rate in enumerate([0.02, 0.1, 0.4]):
            lc, ec = run_repetitions(backup=backup, learning_rate=learning_rate)
            bkp_label = 'Q-learning' if backup == 'q' else 'SARSA'
            plot.add_curve(axid=1, y=lc, cid=i * 3 + j, label=rf'{bkp_label}, $\alpha$ = {learning_rate}')
            plot.add_curve(axid=2, x=greedy_eval_range, y=ec, cid=i * 3 + j)
    plot.add_hline(r_avg_opt, label = 'DP optimum')
    plot.set_ylim(-1.1, 1.5)
    plot.set_titles('Learning curve', 'Greedy evaluation')
    plot.save('on_off_policy.png')
    
    ###### Assignment 4: Back-up depth
    print('\033[1m' + 'Back-up depth' + '\033[0m')
    plot = LearningCurvePlot(title = 'Back-up: depth')    
    for i, n in enumerate([1, 3, 10, 30]):
        lc, ec = run_repetitions(backup='nstep', n=n, n_timesteps=5000)
        plot.add_curve(axid=1, y=lc, cid=i, label=rf'{n}-step Q-learning')
        plot.add_curve(axid=2, x=greedy_eval_range[:10], y=ec, cid=i)
    lc, ec = run_repetitions(backup='mc', n_timesteps=5000)
    plot.add_curve(axid=1, y=lc, cid=4, label='Monte Carlo')
    plot.add_curve(axid=2, x=greedy_eval_range[:10], y=ec, cid=4)
    plot.add_hline(r_avg_opt, label = 'DP optimum')
    plot.set_ylim(-1.1, 1.5)
    plot.set_titles('Learning curve', 'Greedy evaluation')
    plot.save('depth.png')


if __name__ == '__main__':
    np.random.seed(42)
    experiment()
