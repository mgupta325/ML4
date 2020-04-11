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


def extract_policy(env, v, gamma):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma):
    v = np.zeros(env.nS)
    eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v



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


def policy_iteration(env, gamma):
    k=0
    s1=[]
    t2=[]
    i2=[]
    policy = np.random.choice(env.nA, size=(env.nS))
    max_iters = 20000
    desc = env.unwrapped.desc
    for i in range(max_iters):
        st=time.time()
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        et=time.time()
        t2.append(et-st)
        i2.append(i+1)
        # score = evaluate_policy(env, new_policy, gamma)
        s1.append(np.sum(np.fabs(new_policy-policy)))

        print(new_policy.reshape(4,4))
        get_score(env, new_policy)
        plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' Gamma: ' + str(gamma),new_policy.reshape(4,4),desc,colors_lake(),directions_lake())

        if (np.all(policy == new_policy)):
            k = i + 1
            break
        policy = new_policy
    i2=np.array(i2)
    s1=np.array(s1)
    plt.plot(i2, s1)
    plt.xlabel('iterations')
    plt.title('FL- 4 x 4 grid_Policy_Iteration--> convergence')
    plt.ylabel('diff. between curr and prev. policy')
    plt.show()

    t2=np.array(t2)
    print("time taken in policy iteration in frozen lake mdp:",np.mean(t2))

    return policy, k



def value_iteration(env, gamma):
    v1=[]
    t2=[]
    i2=[]
    k=0
    v = np.zeros(env.nS)  # initialize value-function
    max_iters = 20000
    eps = 1e-20
    desc = env.unwrapped.desc
    for i in range(max_iters):
        st=time.time()
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
        et=time.time()
        policy1 = extract_policy(env, v, gamma)
        t2.append(et-st)
        i2.append(i+1)
        v1.append(np.sum(np.fabs(prev_v - v)))
        if i % 50 == 0:
            print(policy1.reshape(4,4))
            get_score(env, policy1)
            plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),policy1.reshape(4,4),desc,colors_lake(),directions_lake())
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            k = i + 1
            break
    i2 = np.array(i2)
    s1 = np.array(v1)
    plt.plot(i2, v1)
    plt.xlabel('iterations')
    plt.title('FL- 4 x 4 grid_Value_Iteration--> convergence')
    plt.ylabel('difference between prev and current value')
    plt.show()

    t2 = np.array(t2)
    print("time in value iteration:", np.mean(t2))

    return v, k


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
    print('Average {:.0f} steps taken to get the frisbee'.format(np.mean(steps_list)))
    print('We fall in the hole {:.2f} % of the times'.format((misses / episodes) * 100))
    print('----------------------------------------------')

environment = 'FrozenLake-v0'   #16 states in grid world mdp
env = gym.make(environment)
env = env.unwrapped
desc = env.unwrapped.desc

###
print('policy iteration on frozen lake mdp')
gamma = 0.95
st = time.time()
best_policy, k = policy_iteration(env, gamma)
scores = evaluate_policy(env, best_policy, gamma)
end = time.time()
plot = plot_policy_map('Frozen Lake Policy Map Iteration ' + ' (PI) ' + 'Gamma: ' + str(gamma),
                       best_policy.reshape(4, 4), desc, colors_lake(), directions_lake())
get_score(env, best_policy)
print("PI time FL",end-st)
print("PI score FL: ", scores)

###
print('value iteration- frozen lake mdp')
gamma = 0.95
st = time.time()
best_value, k = value_iteration(env, gamma)
policy = extract_policy(env, best_value, gamma)
scores = evaluate_policy(env, policy, gamma, n=1000)
end = time.time()
plot = plot_policy_map('Frozen Lake Policy Map Iteration ' + ' (VI) ' + 'Gamma: ' + str(gamma), policy.reshape(4, 4),
                       desc, colors_lake(), directions_lake())
get_score(env, policy)

print("VI time FL",end-st)
print("VI score: ", scores)









