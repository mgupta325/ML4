import gym
from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import mdptoolbox, mdptoolbox.example
import hiive.mdptoolbox.mdp,hiive.mdptoolbox.example


print('Forest management MDP -policy iteration')
P,R=hiive.mdptoolbox.example.forest(S=2000)
t=np.empty(10)
discount=np.empty(10)
iter=np.empty(10)
score=np.empty(10)

for i in range(0, 10):
    pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, (i + 0.5) / 10)
    pi.setVerbose()
    pi.run()
    discount[i] = (i + 0.5) / 10
    score[i] = np.mean(pi.V)
    iter[i] = pi.iter
    t[i] = pi.time
    p=pi.policy
    p=np.array(p)
    print("policy for forest management obtained using PI")
    print(p.reshape(20,100))


plt.plot(discount, t)
plt.xlabel('Discount factor')
plt.title('Policy Iteration_Forest Management - Time ')
plt.ylabel(' Time taken (in seconds)')
plt.grid()
plt.show()


plt.plot(discount, score)
plt.xlabel('Discount factor')
plt.ylabel('Average Rewards')
plt.title('Policy Iteration_Forest Management - Reward ')
plt.grid()
plt.show()

plt.plot(discount, iter)
plt.xlabel('Discount factor')
plt.ylabel('Iterations to Converge')
plt.title('Policy Iteration_Forest Management ')
plt.grid()
plt.show()

print('Forest management MDP -value iteration')
P, R = mdptoolbox.example.forest(S=2000)
t=np.empty(10)
discount=np.empty(10)
iter=np.empty(10)
score=np.empty(10)

for i in range(0, 10):
    pi = hiive.mdptoolbox.mdp.ValueIteration(P, R, (i + 0.5) / 10)
    pi.setVerbose()
    pi.run()
    discount[i] = (i + 0.5) / 10
    score[i] = np.mean(pi.V)
    iter[i] = pi.iter
    t[i] = pi.time
    p=pi.policy
    p=np.array(p)
    print("policy for forest management obtained using VI")
    print(p.reshape(20,100))


plt.plot(discount, t)
plt.xlabel('Discount factor')
plt.title('Value Iteration_Forest Management - Time ')
plt.ylabel('Execution Time (in seconds)')
plt.grid()
plt.show()


plt.plot(discount, score)
plt.xlabel('Discount factor')
plt.ylabel('Average Rewards')
plt.title('Value Iteration_Forest Management - Reward ')
plt.grid()
plt.show()

plt.plot(discount, iter)
plt.xlabel('Discount factor')
plt.ylabel('Iterations to Converge')
plt.title('Value Iteration_Forest Management')
plt.grid()
plt.show()

###tuning done now fixed gamma
pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
pi.run()
print("PI score Fm",np.mean(pi.V))
print("no. of iter",pi.iter)
print("PI time Fm",pi.time)
p = pi.policy
p = np.array(p)
print("policy for forest management obtained using PI")
print(p.reshape(20, 100))

pi = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.95)
pi.run()
print("VI score Fm",np.mean(pi.V))
print("no. of iter",pi.iter)
print("VI time Fm",pi.time)
p = pi.policy
p = np.array(p)
print("policy for forest management obtained using VI")
print(p.reshape(20, 100))




