import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import networkx as nx

class Agent:
    '''
    Q learning agent
    '''
    def __init__(self, gamma, alpha, env):
        self.gamma = gamma
        self.alpha = alpha 
        self.actions = env.action_dict
        Q = {}
        for s in env.action_dict.keys():
            Q[s] = dict([[a, 0] for a in env.action_dict[s]])
        self.Q = Q
        self.update_counts = {} # debugging
        update_counts_sa = {} # adaptive learning rate
        for s, a in self.actions.items(): # gonna use ans a divisor so initialise to 1
            update_counts_sa[s] = dict([[a, 1.0] for a in self.actions[s]])
        self.update_counts_sa = update_counts_sa
        
    def max_dict(self, d):
        '''
        returns the key and value of the dict entry with max value
        '''
        max_key = max(d, key=d.get)
        return max_key, d[max_key]
    

    def random_action(self, env, epsilon):
        s = env.state
        a = self.max_dict(self.Q[s])[0]
        p = np.random.random()
        if p > epsilon:
            return a
        else:
            tmp = list(self.actions[s])
            tmp.remove(a)
            if len(tmp)==0:
                return a
        return np.random.choice(tmp)
    
    def update_Q(self, s, s2, a, r):
        alph = self.alpha / self.update_counts_sa[s][a]
        self.update_counts_sa[s][a] += 0.005
        old_qsa = self.Q[s][a]
        a2, max_q_s2a2 = self.max_dict(self.Q[s2])
        self.Q[s][a] = old_qsa + alph*(r + self.gamma * max_q_s2a2 - old_qsa)
        diff = np.abs(old_qsa - self.Q[s][a])
        self.update_counts[s] = self.update_counts.get(s, 0) + 1
        return diff