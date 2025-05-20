import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode='human' if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array

   

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.001       # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    window_size = 100
    rolling_avg_rewards = np.zeros(episodes)

    for t in range(episodes):
        start = max(0, t - window_size + 1)
        count = t - start + 1
        rolling_avg_rewards[t] = np.mean(rewards_per_episode[start:t+1])  # or sum / count

    plt.plot(rolling_avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Average reward (window={window_size})")
    plt.title("Rolling Average of Episode Rewards")
    plt.show()

if __name__ == '__main__':
    # run(15000)

    run(3000, is_training=True, render=False)