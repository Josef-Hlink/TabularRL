#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self, title: Optional[str] = None) -> None:
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, y: np.ndarray, label: Optional[str] = None) -> None:
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)
    
    def set_ylim(self, lower: float, upper: float) -> None:
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height: float, label: str) -> None:
        self.ax.axhline(height, ls = '--', c = 'k', label = label)

    def save(self, name: str = 'test.png') -> None:
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(f'../plots/{name}', dpi = 500)

def smooth(y: np.ndarray, window: int, poly: int = 1) -> np.ndarray:
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)

def softmax(x: np.ndarray, temp: float) -> np.ndarray:
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z) / np.sum(np.exp(z)) # compute softmax

def argmax(x: np.ndarray) -> int:
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)

def linear_anneal(t: int, T: int, start: float, final: float, percentage: float) -> float:
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    ''' 
    final_from_T = int(percentage * T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01 * x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y,label='method 1')
    LCTest.add_curve(smooth(y,window=35),label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')
