import gym
from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import mdptoolbox, mdptoolbox.example
import hiive.mdptoolbox.mdp,hiive.mdptoolbox.example

print('Q-learning for forest management MDP')
P, R = hiive.mdptoolbox.example.forest(S=2000)
value_f = []
policy = []
iters = []
time_array = []
Q_table = []
rew_array = []
for alpha in [ 0.1, 0.3, 0.5, 0.7,0.85, 0.95]:
    st = time.time()
    pi = hiive.mdptoolbox.mdp.QLearning(P, R,0.95,epsilon=0.7,n_iter=15000,alpha=alpha)

    pi.run()
    end = time.time()

    value_f.append(np.mean(pi.V))
    print(np.mean(pi.V))
    policy.append(pi.policy)
    print("1st 10 states:",pi.policy[0:10])
    print("last 10 states:", pi.policy[1990:2000])
    time_array.append(end - st)
    Q_table.append(pi.Q)
plt.plot(value_f)
plt.title('varying learning rate Q learning Forest Mgmt.')
plt.ylabel('value')
plt.show()
plt.plot(time_array)
plt.title('time vs alpha in Q-learning')
plt.show()



plt.subplot(1, 6, 1)
plt.imshow(Q_table[0][:10, :])
plt.title('alpha=0.1')

plt.subplot(1, 6, 2)
plt.title('alpha=0.3')
plt.imshow(Q_table[1][:10, :])

plt.subplot(1, 6, 3)
plt.title('alpha=0.5')
plt.imshow(Q_table[2][:10, :])

plt.subplot(1, 6, 4)
plt.title('alpha=0.70')
plt.imshow(Q_table[3][:10, :])

plt.subplot(1, 6, 5)
plt.title('alpha=0.85')
plt.imshow(Q_table[4][:10, :])

plt.subplot(1, 6, 6)
plt.title('alpha=0.95')
plt.imshow(Q_table[5][:10, :])
plt.colorbar()
plt.show()

P, R = hiive.mdptoolbox.example.forest(S=2000)
value_f = []
policy = []
iters = []
time_array = []
Q_table = []

for epsilon in [ 0.1, 0.3, 0.5, 0.7,0.85, 0.95]:
    st = time.time()
    pi = hiive.mdptoolbox.mdp.QLearning(P, R,0.95,epsilon=epsilon,n_iter=15000)
    pi.run()
    end = time.time()

    value_f.append(np.mean(pi.V))
    print(np.mean(pi.V))
    policy.append(pi.policy)
    print("1st 10 states:",pi.policy[0:10])
    print("last 10 states:", pi.policy[1990:2000])
    time_array.append(end - st)
    Q_table.append(pi.Q)
plt.plot(value_f)
plt.title('varying epsilon q learning Forest Mgmt.')
plt.ylabel('value')
plt.show()
plt.plot(time_array)
plt.title('time vs epsilon in Q-learning')
plt.show()




plt.subplot(1, 6, 1)
plt.imshow(Q_table[0][:10, :])
plt.title('Epsilon=0.1')

plt.subplot(1, 6, 2)
plt.title('Epsilon=0.3')
plt.imshow(Q_table[1][:10, :])

plt.subplot(1, 6, 3)
plt.title('Epsilon=0.5')
plt.imshow(Q_table[2][:10, :])

plt.subplot(1, 6, 4)
plt.title('Epsilon=0.70')
plt.imshow(Q_table[3][:10, :])

plt.subplot(1, 6, 5)
plt.title('Epsilon=0.85')
plt.imshow(Q_table[4][:10, :])

plt.subplot(1, 6, 6)
plt.title('Epsilon=0.95')
plt.imshow(Q_table[5][:10, :])
plt.colorbar()
plt.show()

P, R = hiive.mdptoolbox.example.forest(S=2000)
value_f = []
policy = []
iters = []
time_array = []
Q_table = []


for gamma in [ 0.1, 0.3, 0.5, 0.7,0.85, 0.95]:
    st = time.time()
    pi = hiive.mdptoolbox.mdp.QLearning(P, R,gamma=gamma,epsilon=0.85,n_iter=15000)

    pi.run()
    end = time.time()

    value_f.append(np.mean(pi.V))
    print(np.mean(pi.V))
    policy.append(pi.policy)
    print("1st 10 states policy:",pi.policy[0:10])
    print("last 10 states policy :", pi.policy[1990:2000])
    time_array.append(end - st)
    Q_table.append(pi.Q)
plt.plot(value_f)
plt.title('varying gamma q learning Forest Mgmt.')
plt.ylabel('value')
plt.show()
plt.plot(time_array)
plt.title('time vs gamma in Q-learning')
plt.show()




plt.subplot(1, 6, 1)
plt.imshow(Q_table[0][:10, :])
plt.title('gamma=0.1')

plt.subplot(1, 6, 2)
plt.title('gamma=0.3')
plt.imshow(Q_table[1][:10, :])

plt.subplot(1, 6, 3)
plt.title('gamma=0.5')
plt.imshow(Q_table[2][:10, :])

plt.subplot(1, 6, 4)
plt.title('gamma=0.70')
plt.imshow(Q_table[3][:10, :])

plt.subplot(1, 6, 5)
plt.title('gamma=0.85')
plt.imshow(Q_table[4][:10, :])
plt.subplot(1, 6, 6)
plt.title('gamma=0.95')
plt.imshow(Q_table[5][:10, :])
plt.colorbar()
plt.show()
##############


print('Q-learning for forest management MDP')

value_f = []
policy = []
time_array = []
Q_table = []

for St in [ 1000,1500,2000,2500,3000]:
    P, R = hiive.mdptoolbox.example.forest(S=St)
    st = time.time()
    pi = hiive.mdptoolbox.mdp.QLearning(P, R,0.95,epsilon=0.8,n_iter=15000,alpha=0.95)
    pi.run()
    end = time.time()

    value_f.append(np.mean(pi.V))
    print(np.mean(pi.V))
    time_array.append(end - st)
    Q_table.append(pi.Q)
    print("1st 10 states:",pi.policy[0:10])
    print("last 10 states:", pi.policy[St-10:St])
plt.plot(value_f)
plt.title('varying number of states in Q learning Forest Mgmt.')
plt.ylabel('value')
plt.show()
plt.plot(time_array)
plt.title('time vs # of states in Q-learning')
plt.show()

###############


print('Q-learning for forest management MDP')

value_f = []
policy = []
iters = []
time_array = []
Q_table = []

for p1 in [ 0.01,.05,.1,.2]:
    P, R = hiive.mdptoolbox.example.forest(S=2000,p=p1)
    st = time.time()
    pi = hiive.mdptoolbox.mdp.QLearning(P, R,0.95,epsilon=0.8,n_iter=10000,alpha=0.95)
    pi.run()
    end = time.time()

    value_f.append(np.mean(pi.V))
    print(np.mean(pi.V))
    time_array.append(end - st)
    Q_table.append(pi.Q)
    print("1st 10 states:",pi.policy[0:10])
    print("last 10 states:", pi.policy[1990:2000])
plt.plot(value_f)
plt.title('varying number of states in Q learning Forest Mgmt.')
plt.ylabel('value')
plt.show()
plt.plot(time_array)
plt.title('time vs # of states in Q-learning')
plt.show()
