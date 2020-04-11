import gym
from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
import time
import sys


def colors_lake():
    return {
            b'S': 'green',
            b'F': 'skyblue',
            b'H': 'black',
            b'G': 'gold',
                 }



def directions_lake():
    return {
            3: '⬆',
            2: '➡',
            1: '⬇',
            0: '⬅'
        }

def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle((x,y), 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.show()
    # plt.savefig(title + str('.jpg'))
    # plt.close()

    return (plt)

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

def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:

            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:

                steps_list.append(steps)
                break
            elif done and reward == 0:

                misses += 1
                break
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(np.mean(steps_list)))
    print('And you fell in the hole {:.2f} % of the times'.format((misses / episodes) * 100))
    print('----------------------------------------------')


environment = 'FrozenLake-v0'   #16 states in grid world mdp
env = gym.make(environment)
env = env.unwrapped
desc = env.unwrapped.desc

print('Frozen Lake-->Q-learning')
Q = np.zeros((env.observation_space.n, env.action_space.n))
Q1 = np.zeros((env.observation_space.n, env.action_space.n))
t=0
alpha = 0.15
gamma = 0.95
epsilon = 0.3
episodes = 12000
rewards=[]
iters=[]
s = time.time()

score1=0
t1=[]
for episode in range(episodes):
    s1=time.time()
    x1=0

    state = env.reset()
    done = False
    t_reward = 0
    max_steps = 200
    new_policy = np.empty(16)

    for i in range(max_steps):

        if done:
            break
        current=state
        if np.random.uniform(0,1) < (epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[current, :])

        state, reward, done, info = env.step(action)
        t_reward += reward
        Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])

    rewards.append(t_reward)
    iters.append(i)
    e1=time.time()
    t1.append(e1-s1)


print("average :", np.average(rewards))
env.close()
end = time.time()
print("time :", end - s)


def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

size = 100
chunks = list(chunk_list(rewards, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]
plt.plot(range(0, len(rewards), size), averages)
plt.xlabel('iters')
plt.ylabel('Average Reward')
plt.title('FL- Q Learning')
plt.show()
# print('iterations to converge',np.argmax(averages)*100+1)

plt.title('Q value array (greedy epsilon) after tuning parameters')
plt.imshow(Q)
plt.colorbar()
plt.show()
print(Q)


policy=np.empty(16)
for state in range(16):
    policy[state] = np.argmax(Q[state, :])
print(policy.reshape(4,4))
scores = evaluate_policy(env, policy, 0.95)
print("final reward is:",scores)
plot = plot_policy_map('Frozen Lake Policy Map (greedy epsilon) ' +' (Q-learning) ' + 'Gamma: ' + str(gamma), policy.reshape(4, 4), desc, colors_lake(), directions_lake())
get_score(env,policy)

######################################

#########################
######################################

environment = 'FrozenLake-v0'   #16 states in grid world mdp
env = gym.make(environment)
env = env.unwrapped
desc = env.unwrapped.desc

print('Frozen Lake-->Q-learning with eps. decay')
Q = np.zeros((env.observation_space.n, env.action_space.n))
Q1 = np.zeros((env.observation_space.n, env.action_space.n))
t=0
alpha = 0.15
gamma = 0.95
epsilon = 0.3
episodes = 12000
rewards=[]
iters=[]
s = time.time()

score1=0
t1=[]
epsilon_min = 0.05
epsilon_decay = 0.99
for episode in range(episodes):
    s1=time.time()
    x1=0

    state = env.reset()
    done = False
    t_reward = 0
    max_steps = 200
    new_policy = np.empty(16)

    for i in range(max_steps):

        if done:
            break
        current=state
        if np.random.uniform(0,1) < (epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[current, :])

        state, reward, done, info = env.step(action)
        t_reward += reward
        Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])

    if episode%300==0:
        epsilon *= epsilon_decay
        if epsilon < epsilon_min:
            epsilon = epsilon_min

    rewards.append(t_reward)
    iters.append(i)
    e1=time.time()
    t1.append(e1-s1)


print("average :", np.average(rewards))
env.close()
end = time.time()
print("time :", end - s)


def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

size = 100
chunks = list(chunk_list(rewards, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]
plt.plot(range(0, len(rewards), size), averages)
plt.xlabel('iters')
plt.ylabel('Average Reward')
plt.title('FL- Q Learning')
plt.show()
# print('iterations to converge',np.argmax(averages)*100+1)

plt.title('Q value array using epsilon decay after tuning parameters')
plt.imshow(Q)
plt.colorbar()
plt.show()
print(Q)


policy=np.empty(16)
for state in range(16):
    policy[state] = np.argmax(Q[state, :])
print(policy.reshape(4,4))
scores = evaluate_policy(env, policy, 0.95)
print("final reward is:",scores)
plot = plot_policy_map('Frozen Lake Policy Map (epsilon decay) ' +' (Q-learning) ' + 'Gamma: ' + str(gamma), policy.reshape(4, 4), desc, colors_lake(), directions_lake())
get_score(env,policy)
