import gymnasium as gym
# Initialise the environment
import numpy as np
import pandas as pd
env = gym.make('Taxi-v3', render_mode="human")

print(env.action_space.n)
print(env.observation_space)

# Initialize parameters
gamma = 0.9  # Discount factor
theta = 1e-6  # Threshold for convergence
num_states = env.observation_space.n  # Number of states
num_actions = env.action_space.n  # Number of actions

# Initialize value function (state values) for each state to zero
V = np.zeros(num_states)

#action space
# 0: Move south (down)

# 1: Move north (up)

# 2: Move east (right)

# 3: Move west (left)

# 4: Pickup passenger

# 5: Drop off passenger

# Passenger locations:

# 0: Red

# 1: Green

# 2: Yellow

# 3: Blue

# 4: In taxi

# Destinations:

# 0: Red

# 1: Green

# 2: Yellow

# 3: Blue

# Helper function to perform value iteration
print(env.action_space.sample())#actions: 0=left , 1 = down, 2 = right, 3 = up)
def value_iteration():
    while True:
        delta = 0
        # Iterate over all states
        for state in range(num_states):
            v = V[state]
            # Bellman update for each state
            action_values = []
            for action in range(num_actions):
                action_value = 0
                # Sum over all possible next states (transitions)
                for next_state, reward, done, _ in env.P[state][action]:
                    action_value += reward + gamma * V[next_state] * (1 - done)
                action_values.append(action_value)
            # Update value function
            V[state] = max(action_values)
            delta = max(delta, abs(v - V[state]))
        
        # If the value function has converged, stop
        if delta < theta:
            break


# q = np.zeros((env.observation_space.n, env.action_space.n))
# learning_rate = 0.8
# discount_factor = 0.9
# episodes = 15000

# epsilon = 1
# epsilon_decay = 0.0001 #1/0.0001 =  10000 episodes to get epsilon down to 0
# rng = np.random.default_rng()

# for i in range(episodes):
#     state = env.reset()[0]
#     terminated = False
#     truncated = False

#     while not terminated and not truncated:
#         if rng.random() < epsilon:#this is so there is some randomness
#             action = env.action_space.sample()#actions: 0=left , 1 = down, 2 = right, 3 = up
#         else:
#             action = np.argmax(q[state,:])
#         new_state, reward, terminated, truncated, info = env.step(action)
#         #print(f"state: {state}, action: {action}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
#         q[state,action] = q[state,action] + learning_rate * (reward + discount_factor * np.max(q[new_state]) - q[state,action])
#         state = new_state
#         if terminated:
#             print(reward)

#     epsilon = max(epsilon - epsilon_decay, 0.01)
#     if(epsilon == 0):
#         learning_rate = 0.0001
#     #create q table to print out
#     df = pd.DataFrame(q,
#                   columns=[f"Action {a}" for a in range(env.action_space.n)],
#                   index=[f"State {s}" for s in range(env.observation_space.n)])
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     pd.set_option('display.max_colwidth', None)  # for older versions of pandas
#     print(df)

env.close()

