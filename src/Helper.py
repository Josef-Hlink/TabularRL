#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self, title: Optional[str] = None) -> None:
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize = (8, 8))
        self.fig.supxlabel('Time')
        self.fig.supylabel('Reward')
        if title is not None:
            self.fig.suptitle(title, fontsize = 16)
        
    def add_curve(self, axid: int, y: np.ndarray, cid: int, x: np.ndarray = None, label: Optional[str] = None) -> None:
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        ax = self.ax1 if axid == 1 else self.ax2
        if x is None: x = np.arange(len(y))
        ax.plot(x, y, color = f'C{cid}', alpha = 0.75, label = label)
    
    def set_ylim(self, lower1: float, upper1: float, lower2: Optional[float] = None, upper2: Optional[float] = None) -> None:
        self.ax1.set_ylim([lower1, upper1])
        if lower2 is not None and upper2 is not None:
            self.ax2.set_ylim([lower2, upper2])
        else:
            self.ax2.set_ylim([lower1, upper1])

    def add_hline(self, height: float, label: str) -> None:
        self.ax1.axhline(height, ls = '--', c = 'k', label = label)
        self.ax2.axhline(height, ls = '--', c = 'k', label = label)

    def set_titles(self, title1: str, title2: str) -> None:
        self.ax1.set_title(title1)
        self.ax2.set_title(title2)

    def save(self, name: str = 'test.png') -> None:
        ''' name: string for filename of saved figure '''
        self.ax1.legend()
        self.fig.tight_layout()
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
