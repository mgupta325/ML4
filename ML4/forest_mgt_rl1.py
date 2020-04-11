import gym
from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import mdptoolbox, mdptoolbox.example
import hiive.mdptoolbox.mdp,hiive.mdptoolbox.example


## after tuning of HP
print('Q-learning for forest management MDP greedy epsilon')

P, R = hiive.mdptoolbox.example.forest(S=2000, p=0.1)
st = time.time()
pi = hiive.mdptoolbox.mdp.QLearning(P, R, 0.95, epsilon=0.8, n_iter=10000,epsilon_decay=1,alpha=0.95)
pi.run()
end = time.time()
print("avg. value",np.mean(pi.V))
print('time',end-st)
print('average action',np.mean(pi.policy))
print("1st 10 states:", pi.policy[0:10])
print("last 10 states:", pi.policy[1990:2000])
print('average action based on mean Q value',np.mean(pi.Q[:,0]),np.mean(pi.Q[:,1]))
plt.imshow(pi.Q[:10,:])
plt.title('Q-array for greedy epsilon Forest mgmt.')
plt.colorbar()
plt.show()




print('Q-learning for forest management MDP decaying epsilon')

P, R = hiive.mdptoolbox.example.forest(S=2000, p=0.1)
st = time.time()
pi = hiive.mdptoolbox.mdp.QLearning(P, R, 0.95, epsilon=0.8,epsilon_decay=0.99, n_iter=10000, alpha=0.95)
pi.run()
end = time.time()
print("avg. value",np.mean(pi.V))
print('time',end-st)
print('average action',np.mean(pi.policy))
print("1st 10 states:", pi.policy[0:10])
print("last 10 states:", pi.policy[1990:2000])
print('average action based on mean Q value',np.mean(pi.Q[:,0]),np.mean(pi.Q[:,1]))
plt.imshow(pi.Q[:10,:])
plt.title('Qarray for decaying epsilon Forest mgmt.')
plt.colorbar()
plt.show()
