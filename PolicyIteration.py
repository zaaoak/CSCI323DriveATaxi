from collections import defaultdict
import gymnasium as gym
# Initialise the environment
import numpy as np
import pandas as pd
from typing import Callable, Optional, Self, Tuple
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
import time
from itertools import product

import tkinter as tk
import pandas as pd
from itertools import product
from typing import Tuple
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

env = gym.make('Taxi-v3', render_mode="ansi")

num_states = env.observation_space.n  # 500
num_actions = env.action_space.n       # 6 actions in Taxi

# # Dictionary to hold action masks per state
# state_action: dict[Tuple[int,int,int,int],dict] = defaultdict(dict)#each element is 'action: (obs, reward, done, truncated, info)'
# #NOTE: drop off is always legal if it is dropped off at a wrong hotel, the state changes(not terminal) and time step penalty of -1 happens, but drop off in the middle of the street, is illegal, state doesnt change, and penalty is -10
# for state in range(num_states):
#     env.reset()
#     env.unwrapped.s = state  # Manually set state
#     originState = decode_state(state)
#     #print(env.unwrapped.s)
    
#     #IMPORTANT NOTE, gymnasium allows drop off at other locations, it is LEGAL, the only ILLEGAL drop off and state doesnt change, is anywhere outside the 4 hotels
#     # if (decode_state(state)==(0,0,4,1)):
#     #     obs, reward, done, truncated, info = env.step(5)
#     #     print(f' obs {decode_state(obs)} action mask: {info.get("action_mask")}')
#     #     print(f'reward: {reward}, done: {done}, truncated: {truncated}, info: {info}')

#     #go through all actions
#     for action in range(num_actions):
#         newState, reward, done, truncated, info = env.step(action)
#         #determine, for this action, what is the resulting state and reward
#         state_action[originState][action] = (decode_state(newState), reward, done, truncated, info)
#         #print(f' obs {decode_state(newState)} action mask: {info.get("action_mask")}')
#         #print(f'reward: {reward}, done: {done}, truncated: {truncated}, info: {info}')
       

#set a fixed passenger start location and hotel destination
passengerStart = 0
Destination = 1
#truncation limit
env = TimeLimit(env.unwrapped, max_episode_steps=200)
#====================
env.reset()
# Function to encode state from components
def encode_state( taxi_col, taxi_row,passenger_loc, destination):
    return ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination
# Desired initial state: taxi at (0, 0), passenger at location 0 (R), destination at 1 (G)
initial_state = encode_state(taxi_row=0, taxi_col=0, passenger_loc=passengerStart, destination=Destination)
# Set the environment's internal state
env.unwrapped.s = initial_state

epsilon = 1e-5
maxEpsilon = 1
discount = 0.4

#initialise state value  stable
x_range = range(5)         # 0 to 4
y_range = range(5)         # 0 to 4
passenger_values = [0,2,3, 4]  # we have to include all passenger locations, even though we only want pasenger at 1, because dropping off passenger at wrong hotel is allowed to change state, although get reward of -10
dest_values = [1]          # only 1

combinations = list(product(x_range, y_range, passenger_values, dest_values))
VTable:dict[Tuple[int,int,int,int],int] ={item: 0 for item in combinations}#V(s) for each state
PolicyTable = {item: 0 for item in combinations}#state : Action



# # Build GUI
# def update_display():
#     df = pd.DataFrame({
#         'State': list(VTable.keys()),
#         'V(s)': list(VTable.values()),
#         'Policy (Action)': [PolicyTable[state] for state in VTable]
#     })
#     text_widget.delete('1.0', tk.END)
#     text_widget.insert(tk.END, df.to_string(index=False))
#     root.after(1000, update_display)  # refresh every 1 second

# # --- Setup Tkinter GUI ---
# root = tk.Tk()
# root.title("Live VTable and PolicyTable Display")

# # Define the text widget here!
# text_widget = tk.Text(root, wrap='none', font=("Courier", 9))
# text_widget.pack(expand=True, fill='both')

# # Start updating the display
# update_display()

# # Run the GUI event loop
# root.mainloop()


converged = False
iterations = 0
steps = 0
while not converged:
    
    iterations+=1
    #iteration step
    epsilonThreshhold = 1e-5
    while epsilon < maxEpsilon:
        epsilonTab = []
        #evaluate policy step
        for x_taxi, y_taxi, passenger, dest in combinations:
            env.reset()
            env.unwrapped.s = encode_state(x_taxi, y_taxi, passenger, dest)
            actionToTake = PolicyTable[(x_taxi, y_taxi, passenger, dest)]
            newState, reward, done,truncated, info = env.step(actionToTake)
            futureV = 0 if done else VTable[decode_state(newState)]
            v = reward + discount * futureV
            epsilonTab.append(abs(VTable[(x_taxi, y_taxi, passenger, dest)] - v))
            VTable[(x_taxi, y_taxi, passenger, dest)] = v
            
        #calculate epsilon
        max_epsilon = max(epsilonTab)
        print(f'max epsilon: {max_epsilon}')
        steps +=1
        if max_epsilon < epsilonThreshhold:
            print("======================================== policy evaluation converged, moving on to policy improvement")
            break
    converged = True
    #policy improvement
    for x_taxi, y_taxi, passenger, dest in combinations:
        QValues = [0,0,0,0,0,0]
        for action in range(6):
            env.reset()
            env.unwrapped.s = encode_state(x_taxi, y_taxi, passenger, dest)
            newState, reward, done,truncated, info = env.step(action)
            futureV = 0 if done else VTable[decode_state(newState)]
            QValues[action] = reward + discount * futureV
        
        best_action = QValues.index(max(QValues))
        current_action = PolicyTable[(x_taxi, y_taxi, passenger, dest)]
        if current_action != best_action:
            converged = False
            PolicyTable[(x_taxi, y_taxi, passenger, dest)] = best_action
       # print(x_taxi, y_taxi, passenger, dest, best_action)

print(f"Iterations: {iterations}, Steps: {steps}, converged!")





