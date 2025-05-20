import gymnasium as gym
# Initialise the environment
import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
import time
action_names = {
    0: "down",
    1: "up",
    2: "right",
    3: "left",
    4: "Pickup",
    5: "DropOff"
}
def decode_state(state: int):
    dest = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    taxi_col = state % 5
    state //= 5
    taxi_row = state % 5

    return taxi_col,taxi_row, passenger, dest#(x, y, passenger, destination)
def print_q_table_filtered(
    q: np.ndarray,
    env,
    filter_passenger_destination: Tuple[Optional[int], Optional[int]],
    decode_state: Callable[[int], Tuple[int, int, int, int]] = decode_state
):
    """
    Print the Q-table, filtering by passenger and destination values.
    
    Parameters:
    - q: Q-table (NumPy array)
    - env: Gymnasium environment
    - decode_state: Function that returns (taxi_row, taxi_col, passenger, destination)
    - filter_passenger_destination: Tuple of (passenger, destination) to filter
    
    Special rule: always include passenger == 4 (in taxi)
    """

    
    data = []
    passenger_target, destination_target = filter_passenger_destination
    
    for state in range(env.observation_space.n):
        taxi_col, taxi_row, passenger, destination = decode_state(state)

        include = (
            (passenger_target is None or passenger == passenger_target) and
            (destination_target is None or destination == destination_target)
        )

        if include or (passenger == 4 and destination == destination_target):
            q_values = q[state]
            best_action = int(np.argmax(q_values))
            best_q_value = q_values[best_action]
            
            data.append({
                "state": state,
                "taxi_row": taxi_row,
                "taxi_col": taxi_col,
                "passenger": passenger,
                "destination": destination,
                "q_values": q_values,
                "best_action": best_action,
                "best_q_value": best_q_value,
            })

    data.sort(key=lambda x: (x["passenger"], x["taxi_col"], x["taxi_row"], x["destination"]))

    rows = [item["q_values"] for item in data]
    best_actions = [action_names.get(item["best_action"], f"Action {item['best_action']}") for item in data]
    best_qs = [item["best_q_value"] for item in data]
    row_labels = [
        f"Taxi({item['taxi_col']},{item['taxi_row']}) P:{item['passenger']} D:{item['destination']}"
        for item in data
    ]

    # Use descriptive column names for actions
    columns = [action_names.get(a, f"Action {a}") for a in range(env.action_space.n)]

    df = pd.DataFrame(
        rows,
        columns=columns,
        index=row_labels
    )
    df["Best Action"] = best_actions
    df["Best Q-Value"] = best_qs

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df)


env = gym.make('Taxi-v3', render_mode="ansi")

q = np.zeros((env.observation_space.n, env.action_space.n))

#environment setup============
learning_rate = 0.8
discount_factor = 0.9

#for epsilon greedy, 1 represents pure exploration(random action taken every step), 0 represents pure exploitation(only choose best action every step)
epsilon = 1
epsilon_decay = 0.001 #1/0.0001 =  10000 episodes to get epsilon down to 0
rng = np.random.default_rng()

pureExploit = False
singleStartState = True
#truncation limit
env = TimeLimit(env.unwrapped, max_episode_steps=200)
#====================

env.reset()
# Function to encode state from components
def encode_state(taxi_col,taxi_row, passenger_loc, destination):
    return ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination

# Desired initial state: taxi at (0, 0), passenger at location 0 (R), destination at 1 (G)
initial_state = encode_state(taxi_row=0, taxi_col=0, passenger_loc=0, destination=1)

# Set the environment's internal state
env.unwrapped.s = initial_state
# Render to verify
#env.render()
episodes = 10000
printLastFewEpisodes = 5
printLastFewEpisodesButOnlyUnOptimal = True
rewards = []#for keeping track of rewards at each episode
optimalAction = {
    (0,0,0,1):4,
    (0,0,4,1):0,
    (0,1,4,1):0,
    (0,2,4,1):2,
    (1,2,4,1):2,
    (2,2,4,1):1,
    (2,1,4,1):1,
    (2,0,4,1):2,
    (3,0,4,1):2,
    (4,0,4,1):5,
}

start_time = time.time()
for i in range(episodes):
    env.reset()
    env.unwrapped.s = initial_state
    state = initial_state
    if not singleStartState:
        state = env.reset()[0]#random start state
    terminated = False
    truncated = False
    if (i >episodes -printLastFewEpisodes):
        print(f"start state {decode_state(state)} ")
    totalReward = 0
    randomact = 0#number of random actions taken(exploration)
    bestact = 0#number of best actions taken(exploitation)
    actions = []
    qDiff = []
    while not terminated and not truncated:
        action = 0
        if pureExploit:
            action = np.argmax(q[state,:])
        else:
            if rng.random() < epsilon:#this is so there is some randomness
                randomact += 1
                #do a random action
                action = env.action_space.sample()#actions: 0=left , 1 = down, 2 = right, 3 = up
            else:
                bestact +=1
                #take best action according to current q table
                action = np.argmax(q[state,:])
        # #if epsilon is 0 already, and for some reason model takes unoptimal step, show current q table:
        # if epsilon == 0.01 and decode_state(state) in optimalAction and action != optimalAction[decode_state(state)]:
        #     print(f"random actions taken: {randomact}, best actions taken: {bestact}, not sure why its going unoptimally, pringing q table")
        #     print_q_table_filtered(q, env, (0,1))

        new_state, reward, terminated, truncated, info = env.step(action)
        qNew = q[state,action] + learning_rate * (reward + discount_factor * np.max(q[new_state]) - q[state,action])
        delta = (np.abs(qNew - q[state,action]))#to track the delta of each change
        q[state,action] = qNew
        qDiff.append(delta)
        totalReward += reward
        actions.append(action_names[action])
        state = new_state
        if terminated or truncated:
            if (i >episodes -printLastFewEpisodes):
                print(f'last action taken: {action},current state {decode_state(state)} ,truncated: {truncated}, terminated: {terminated}')
                print(f"episode {i} ended rewards collected: {totalReward}, current epsilon: {epsilon},current learning rate: {learning_rate}")
                print(f"random actions taken: {randomact}, best actions taken: {bestact}, actions: {actions}")
                print_q_table_filtered(q, env, (0,1))
                print("========================================================\n")
    rewards.append(totalReward)
    #print(f'max delta : {max(qDiff)}')

    #print_q_table_filtered(q, env, (0,1))
    epsilon = epsilon - epsilon_decay
    #print(f'max qdiff: {max(qDiff)}')
    # def Maverage(arr, window_size=episodes/10):
    #     window = arr[-window_size:]  # last 100 or fewer elements
    #     return sum(window) / len(window)
    # if(Maverage(rewards) ==11):
    #     print(f'Threshold reached, function converged')
    #     break
    # if(max(qDiff) < thresHold):
    #     print(f'Threshold reached, function converged')
    #     print(f"episode {i} ended rewards collected: {totalReward}, current epsilon: {epsilon},current learning rate: {learning_rate}")
    #     print(f"random actions taken: {randomact}, best actions taken: {bestact}, actions: {actions}")
    #     print_q_table_filtered(q, env, (0,1))
    #     print("========================================================\n")
    #     break
env.close()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")
def moving_average(data, window_size=episodes/10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid') 
window_size = 100  # Average over every 100 episodes
avg_rewards = moving_average(rewards, window_size)

#print(env.spec.max_episode_steps)
plt.plot(avg_rewards)
plt.xlabel('Episode (offset by window size)')
plt.ylabel(f'Average Reward (over {episodes} episodes)'.format(window_size))
plt.title('Moving Average of Rewards')
plt.grid(True)
plt.show()
