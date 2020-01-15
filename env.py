import random
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import networkx as nx

class Corridor:
    '''
    bla bla
    '''
    def __init__(self, branches, stepcost, branchlength, context_dict):
        self.branches = branches
        self.pc = context_dict
        self.stepcost = stepcost # keys are tuples (k, m), k = reward location, m = indicator obs
        self.state = None
        self.episode_states = None
        self.all_states = None
        self.reward_loc = None
        self.L = branchlength
        self.initialized = False
        # enumeration of locations is [ij], i = branch number, j = distance from central vertex. Starting branch is 0
        # states are represented as string, ijklm, ij = loc, k = reward loc, l = reward obs, m = inidicator obs
        
        # create list of states
        # [0,0] is unique middle state. Only one state begins in 0
        # [i,0] is first state on given branch
        # [i,j] is jth state along ith branch
        all_states = []
        for i in range(branches + 1):
            for j in range(branchlength+1):
                # single state begging with 0 - middle state
                if (j==0) & (i>0):
                    continue
                # location label
                position = f'{i}{j}'
                for k in range(1, branches+1):
                    reward_loc = str(k)
                    if (i == k) & (j == self.L): # if the agent is located at a reward, we observe this
                        l = 1
                    else:
                        l = 0
                    partial_state = position + reward_loc + str(l)
                    for m in range(3):
                        # append indicator variable onto state
                        # 0 = unobserved, 1 = observed in state 1, 2 = observed in state 2
                        if (i==0) & (j==self.L) & (m==0):
                            continue
                        final_state = partial_state + str(m) 
                        all_states.append(final_state)
        self.all_states = all_states  
        
        # create state-action dictionary
        # terminal states have action 'None'
        # actions are +1, -1 for advancing or moving back along a coridor
        # in the middle coridor, the action is to choose a coridor 
        action_dict = {}
        for state in self.all_states:
            if (int(state[-2]) == 1): # observing reward = terminal state
                action_dict[state] = [None]
            elif state[:2] == '00':
                action_dict[state] = [i for i in range(branches + 1)]
            elif int(state[1])!=branchlength:
                action_dict[state] = [1, -1]
            else:
                action_dict[state] = [-1]
                
        self.action_dict = action_dict
        
    def is_terminal(self):
        return int(self.state[-2]) == 1 
    
    def set_state(self, s):
        self.state = s
        pass
    
    def initialize_episode(self):
        # sample context distribution
        x = random.uniform(0, 1)
        cum = 0
        for k, val in self.pc.items():
            cum += val
            if x <= cum:
                context = k
                break
        k, l = context
        # inital state
        self.state = f'01{k}0{0}' # initialize state with no observation
        self.indicator = l
        self.episode_states = [s for s in self.all_states if (s[2]==k)&(s[4]==l)]
        self.reward_loc = k
        self.initialized = True
        pass

    def move(self, action):
        state = self.state
        r = self.stepcost
        if action not in self.action_dict[state]:
            print('Error: action invalid')
        else:
            if state[:2] == '00':
                new_state = f'{action}1' + state[2:]
            elif (action == -1) & (int(state[1]) == 1):
                # moving back to middle
                new_state = '00' + state[2:]
            else:
                new_pos = int(state[1])+action
                # was observation of indicator made?
                if (new_pos == self.L) & (int(state[0]) == 0):
                    ind = self.indicator
                    new_state = state[0] + f'{new_pos}' + state[2:-1] + str(ind)
                elif (new_pos == self.L) & (state[0] == self.reward_loc): # reward never in indicator corridor
                    new_state = state[0] + f'{new_pos}' + state[2] + '1' + state[4]
                    r = 1
                else:
                    new_state = state[:1] + f'{new_pos}' + state[2:]
        self.state = new_state
        return new_state, r

    def game_over(self):
        return  self.is_terminal(self.state)     


    def print_env(self):
        if not self.initialized:
            print('Initialize environment')
            pass
        L = self.L
        N = self.branches
        G = nx.Graph()
        #G.add_node("a")
        G.add_nodes_from(list(set([i[:2] for i in self.all_states])))

        # add edges from vertex
        for i in range(N+1):
            edge = ("00", f"{i}1")
            G.add_edge(*edge)
        for i in range(N + 1):
            for j in range(1, L):
                edge = (f"{i}{j}", f"{i}{j+1}")
                G.add_edge(*edge)
        loc_a = self.state[:2]
        loc_r = self.reward_loc + f'{L}'
        colour_map = []
        for node in G:
            if node == loc_a:
                colour_map.append('green')
            elif node == loc_r: 
                colour_map.append('orange') 
            else:
                colour_map.append('blue')
        nx.draw_spring(G,node_color = colour_map, with_labels=True)
        plt.show()
        pass
