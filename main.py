from agent import Agent
from env import Corridor
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import networkx as nx

if __name__=='__main__':
	gamma = 0.9
	alpha = 0.1
	N = 3
	L = 2
	epochs = 10000
	context_dict = {'11': 0.15/2,
	            '12': 0.85/2,
	            '21': 0.85/2,
	            '22': 0.05/2,
	            '31': 0.05/2,
	            '32': 0.15/2
	           }
	env = Corridor(N, -0.2, L, context_dict)
	agent = Agent(gamma, alpha, env)


	t = 1.0
	deltas = []
	for it in range(epochs):
		if it % 1000 == 0: # updating t used for decay rate in epsilon greedy
			t += 1#0e-3

		env.initialize_episode()
		biggest_change = 0
		# prints every step of env
		#print_flag = (it % 2000 == 0)
		#if print_flag == 1:
	  	#  print('Epoch = %.3f' % it)
		#movecount = 0
		while not env.is_terminal():
			#if print_flag == 1:
			if it == epochs - 1:
				env.print_env()
				#print('_____________________________________________________')
			s = env.state
			a = agent.random_action(env, epsilon = 0.5/t)
			s2, r = env.move(a)
			diff = agent.update_Q(s, s2, a, r)    
			biggest_change = max(biggest_change, diff)
			#movecount +=1
			#print(movecount)
		deltas.append(biggest_change)
	plt.plot(deltas)
	plt.show()


