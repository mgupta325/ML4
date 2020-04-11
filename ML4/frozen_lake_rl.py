import gym
from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

def run_episode(env, policy, gamma, render=True):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma, n=500):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

environment = 'FrozenLake-v0'   #16 states in grid world mdp
env = gym.make(environment)
env = env.unwrapped
desc = env.unwrapped.desc



print('Frozen Lake-->Q-learning')

reward_array = []
r_array=[]
iter_array = []
size_array = []
ts_array = []
averages_array = []
averages_array1 = []
time_array = []
Q_array = []
for epsilon in [ 0.3,0.5, 0.7,0.8, 0.9, 0.95]:
    st = time.time()
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iters = []
    optimal = [0] * env.observation_space.n
    alpha = 0.85
    gamma = 0.95
    episodes = 8000
    environment = 'FrozenLake-v0'
    env = gym.make(environment)
    env = env.unwrapped
    desc = env.unwrapped.desc
    for episode in range(episodes):
        state = env.reset()
        done = False
        t_reward = 0
        max_steps = 200
        for i in range(max_steps):
            if done:
                break
            current = state
            if np.random.uniform(0, 1) < (epsilon):
                action = env.action_space.sample()

            else:
                action = np.argmax(Q[current, :])

            state, reward, done, info = env.step(action)
            t_reward += reward
            Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
        # epsilon = (1 - 2.71 ** (-episode / 1000))    #decaying epsilon
        rewards.append(t_reward)
        iters.append(i)

    for k in range(env.observation_space.n):
        optimal[k] = np.argmax(Q[k, :])

    reward_array.append(rewards)
    r1=np.array(rewards)
    r_array.append(np.mean(r1))
    iters = np.array(iters)
    iter_array.append(np.sum(iters)/episodes)
    Q_array.append(Q)

    env.close()
    end = time.time()
    # print("time :",end-st)
    time_array.append(end - st)


    # Plot results
    def t_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


    size = int(episodes / 50)
    ts = list(t_list(rewards, size))
    averages = [sum(t) / len(t) for t in ts]
    size_array.append(size)
    ts_array.append(ts)
    averages_array.append(averages)
    averages_array1.append(np.mean(averages))

print(Q_array[0],Q_array[1],Q_array[2],Q_array[3],Q_array[4],Q_array[5])

plt.plot([ 0.3,0.5,  0.7, 0.8,0.9, 0.95], time_array)
plt.xlabel('epsilon Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Execution Time (s)')
plt.show()

plt.plot([ 0.3,0.5,  0.7, 0.8,0.9, 0.95], iter_array)
plt.xlabel('epsilon Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Iterations')
plt.show()

plt.plot([ 0.3,0.5,  0.7,0.8, 0.9, 0.95], averages_array1)
plt.xlabel('epsilon Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Rewards')
plt.show()

plt.subplot(1, 4, 1)
plt.imshow(Q_array[0])
plt.title('epsilon=0.1')

plt.subplot(1, 4, 2)
plt.title('epsilon=0.5')
plt.imshow(Q_array[2])

plt.subplot(1, 4, 3)
plt.title('epsilon=0.7')
plt.imshow(Q_array[3])

plt.subplot(1, 4, 4)
plt.title('epsilon=0.9')
plt.imshow(Q_array[4])
plt.colorbar()
plt.show()



plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='epsilon=0.5')
plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='epsilon=0.7')
plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='epsilon=0.95')

plt.legend()
plt.xlabel('Epsiodes')
plt.grid()
plt.title('Frozen Lake - Q Learning - Constant epsilon')
plt.ylabel('Average Reward')
plt.show()

print('Frozen Lake-->Q-learning varying gamma')

reward_array = []
r_array=[]
iter_array = []
size_array = []
ts_array = []
averages_array = []
averages_array1 = []
time_array = []
Q_array = []
for gamma in [0.1, 0.3,0.5,  0.7, 0.9, 0.95]:
    st = time.time()
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iters = []
    optimal = [0] * env.observation_space.n
    alpha = 0.8
    epsilon = 0.3
    episodes = 8000
    environment = 'FrozenLake-v0'
    env = gym.make(environment)
    env = env.unwrapped
    desc = env.unwrapped.desc
    for episode in range(episodes):
        state = env.reset()
        done = False
        t_reward = 0
        max_steps = 200
        for i in range(max_steps):
            if done:
                break
            current = state
            if np.random.uniform(0, 1) < (epsilon):
                action = env.action_space.sample()

            else:
                action = np.argmax(Q[current, :])

            state, reward, done, info = env.step(action)
            t_reward += reward
            Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
        # epsilon = (1 - 2.71 ** (-episode / 1000))    #decaying epsilon
        rewards.append(t_reward)
        iters.append(i)

    for k in range(env.observation_space.n):
        optimal[k] = np.argmax(Q[k, :])

    reward_array.append(rewards)
    r1=np.array(rewards)
    r_array.append(np.mean(r1))
    iters = np.array(iters)
    iter_array.append(np.sum(iters)/episodes)
    Q_array.append(Q)

    env.close()
    end = time.time()
    # print("time :",end-st)
    time_array.append(end - st)


    # Plot results
    def t_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


    size = int(episodes / 50)
    ts = list(t_list(rewards, size))
    averages = [sum(t) / len(t) for t in ts]
    size_array.append(size)
    ts_array.append(ts)
    averages_array.append(averages)
    averages_array1.append(np.mean(averages))

print(Q_array[0],Q_array[1],Q_array[2],Q_array[3],Q_array[4],Q_array[5])

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], time_array)
plt.xlabel('gamma Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Execution Time (s)')
plt.show()

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], iter_array)
plt.xlabel('gamma Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Iterations')
plt.show()

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], averages_array1)
plt.xlabel('gamma Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Rewards')
plt.show()

plt.subplot(1, 4, 1)
plt.imshow(Q_array[0])
plt.title('gamma=0.1')

plt.subplot(1, 4, 2)
plt.title('gamma=0.5')
plt.imshow(Q_array[2])

plt.subplot(1, 4, 3)
plt.title('gamma=0.7')
plt.imshow(Q_array[3])

plt.subplot(1, 4, 4)
plt.title('gamma=0.9')
plt.imshow(Q_array[4])
plt.colorbar()
plt.show()



plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='gamma=0.5')
plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='gamma=0.7')
plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='gamma=0.95')

plt.legend()
plt.xlabel('Epsiodes')
plt.grid()
plt.title('Frozen Lake - Q Learning - Constant gamma')
plt.ylabel('Average Reward')
plt.show()



print('Frozen Lake-->Q-learning varying alpha ')

reward_array = []
r_array=[]
iter_array = []
size_array = []
ts_array = []
averages_array = []
averages_array1 = []
time_array = []
Q_array = []
for alpha in [0.1, 0.3,0.5,  0.7, 0.9, 0.95]:
    st = time.time()
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iters = []
    optimal = [0] * env.observation_space.n
    gamma = 0.95
    epsilon = 0.3
    episodes = 8000
    environment = 'FrozenLake-v0'
    env = gym.make(environment)
    env = env.unwrapped
    desc = env.unwrapped.desc
    for episode in range(episodes):
        state = env.reset()
        done = False
        t_reward = 0
        max_steps = 200
        for i in range(max_steps):
            if done:
                break
            current = state
            if np.random.uniform(0, 1) < (epsilon):
                action = env.action_space.sample()

            else:
                action = np.argmax(Q[current, :])

            state, reward, done, info = env.step(action)
            t_reward += reward
            Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
        # epsilon = (1 - 2.71 ** (-episode / 1000))    #decaying epsilon
        rewards.append(t_reward)
        iters.append(i)

    for k in range(env.observation_space.n):
        optimal[k] = np.argmax(Q[k, :])

    reward_array.append(rewards)
    r1=np.array(rewards)
    r_array.append(np.mean(r1))
    iters = np.array(iters)
    iter_array.append(np.sum(iters)/episodes)
    Q_array.append(Q)

    env.close()
    end = time.time()
    # print("time :",end-st)
    time_array.append(end - st)


    # Plot results
    def t_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


    size = int(episodes / 50)
    ts = list(t_list(rewards, size))
    averages = [sum(t) / len(t) for t in ts]
    size_array.append(size)
    ts_array.append(ts)
    averages_array.append(averages)
    averages_array1.append(np.mean(averages))

print(Q_array[0],Q_array[1],Q_array[2],Q_array[3],Q_array[4],Q_array[5])

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], time_array)
plt.xlabel('alpha Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Execution Time (s)')
plt.show()

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], iter_array)
plt.xlabel('alpha Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Iterations')
plt.show()

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], averages_array1)
plt.xlabel('alpha Values')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Rewards')
plt.show()

plt.subplot(1, 4, 1)
plt.imshow(Q_array[0])
plt.title('alpha=0.1')

plt.subplot(1, 4, 2)
plt.title('alpha=0.5')
plt.imshow(Q_array[2])

plt.subplot(1, 4, 3)
plt.title('alpha=0.7')
plt.imshow(Q_array[3])

plt.subplot(1, 4, 4)
plt.title('alpha=0.9')
plt.imshow(Q_array[4])
plt.colorbar()
plt.show()



plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='alpha=0.5')
plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='alpha=0.7')
plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='alpha=0.95')

plt.legend()
plt.xlabel('Epsiodes')
plt.grid()
plt.title('Frozen Lake - Q Learning - Constant alpha')
plt.ylabel('Average Reward')
plt.show()

# r_array=[]
# reward_array = []
# iter_array = []
# size_array = []
# chunks_array = []
# averages_array = []
# time_array = []
# Q_array = []
# for epsilon in [0.3, 0.5, 0.7, 0.85,0.9,.99]:
#     st = time.time()
#     Q = np.zeros((env.observation_space.n, env.action_space.n))
#     rewards = []
#     iters = []
#     optimal = [0] * env.observation_space.n
#     alpha = 0.8
#     gamma = 0.95
#     episodes = 10000
#     environment = 'FrozenLake-v0'
#     env = gym.make(environment)
#     env = env.unwrapped
#     desc = env.unwrapped.desc
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         t_reward = 0
#         max_steps = 200
#         for i in range(max_steps):
#             if done:
#                 break
#             current = state
#             if np.random.uniform(0, 1) < (epsilon):
#                 action = env.action_space.sample()
#
#             else:
#                 action = np.argmax(Q[current, :])
#
#             state, reward, done, info = env.step(action)
#             t_reward += reward
#             Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
#         # if episode%1000==0:
#         #     epsilon = epsilon*0.95 #decaying epsilon
#
#         rewards.append(t_reward)
#         iters.append(i)
#     # print("nw ep:", epsilon)
#     # def chunk_list(l, n):
#     #     for i in range(0, len(l), n):
#     #         yield l[i:i + n]
#     # size = 40
#     # chunks = list(chunk_list(rewards, size))
#     # averages = [sum(chunk) / len(chunk) for chunk in chunks]
#     # plt.plot(range(0, len(rewards), size)[0:150:5], averages[0:150:5])
#     # plt.xlabel('iters')
#     # plt.ylabel('Average Reward')
#     # plt.title('FL- Q Learning '+str(epsilon))
#     # plt.show()
#     print("average :", np.average(rewards[1000:]))
#
#     policy = np.empty(16)
#     for state in range(16):
#         policy[state] = np.argmax(Q[state, :])
#     print(policy.reshape(4, 4))
#     scores = evaluate_policy(env, policy, 0.95)
#     print("final reward is:", scores)
#
#     reward_array.append(np.average(rewards[1000:]))
#
#     Q_array.append(Q)
#
#     env.close()
#     end = time.time()
#     # print("time :",end-st)
#     time_array.append(end - st)
#
#     print("time:",end-st)
#
#
# plt.plot([0.05, 0.1, 0.3, 0.5, 0.7, 0.9], time_array)
# plt.xlabel('Epsilon Values')
# plt.grid()
# plt.title('Frozen Lake - Q Learning(time)')
# plt.ylabel('Execution Time (s)')
# plt.show()
#
#
# plt.plot([0.05, 0.1, 0.3, 0.5, 0.7, 0.9], reward_array)
# plt.xlabel('Epsilon Values')
# plt.grid()
# plt.title('Frozen Lake - Q Learning(rewards)')
# plt.ylabel('Rewards')
# plt.show()
#
# plt.subplot(1, 6, 1)
# plt.imshow(Q_array[0])
# plt.title('Epsilon=0.05')
#
# plt.subplot(1, 6, 2)
# plt.title('Epsilon=0.1')
# plt.imshow(Q_array[1])
#
# plt.subplot(1, 6, 3)
# plt.title('Epsilon=0.3')
# plt.imshow(Q_array[2])
#
# plt.subplot(1, 6, 4)
# plt.title('Epsilon=0.50')
# plt.imshow(Q_array[3])
#
# plt.subplot(1, 6, 5)
# plt.title('Epsilon=0.7')
# plt.imshow(Q_array[4])
#
# plt.subplot(1, 6, 6)
# plt.title('Epsilon=0.85')
# plt.imshow(Q_array[5])
# plt.colorbar()
#
# plt.show()

