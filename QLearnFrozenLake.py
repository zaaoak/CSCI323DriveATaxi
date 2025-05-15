import gymnasium as gym
# Initialise the environment
import numpy as np
import pandas as pd
env = gym.make('FrozenLake-v1',map_name="8x8",is_slippery=True, render_mode=None)

q = np.zeros((env.observation_space.n, env.action_space.n))
learning_rate = 0.8
discount_factor = 0.9
episodes = 15000

epsilon = 1
epsilon_decay = 0.0001 #1/0.0001 =  10000 episodes to get epsilon down to 0
rng = np.random.default_rng()

for i in range(episodes):
    state = env.reset()[0]
    terminated = False
    truncated = False

    while not terminated and not truncated:
        if rng.random() < epsilon:#this is so there is some randomness
            action = env.action_space.sample()#actions: 0=left , 1 = down, 2 = right, 3 = up
        else:
            action = np.argmax(q[state,:])
        new_state, reward, terminated, truncated, info = env.step(action)
        #print(f"state: {state}, action: {action}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
        q[state,action] = q[state,action] + learning_rate * (reward + discount_factor * np.max(q[new_state]) - q[state,action])
        state = new_state
        if terminated:
            print(reward)

    epsilon = max(epsilon - epsilon_decay, 0.01)
    if(epsilon == 0):
        learning_rate = 0.0001
    #create q table to print out
    df = pd.DataFrame(q,
                  columns=[f"Action {a}" for a in range(env.action_space.n)],
                  index=[f"State {s}" for s in range(env.observation_space.n)])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)  # for older versions of pandas
    print(df)

env.close()

