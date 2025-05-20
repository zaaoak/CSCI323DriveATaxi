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
       

class TaxiStateValueRow:
    discount = 0.8
    def __init__(self, x_taxi, y_taxi, passenger, dest, v_s,
                 q_down, q_up, q_right, q_left, q_pickup, q_dropoff,
                 best_action, epsilon):
        self.x_taxi = x_taxi
        self.y_taxi = y_taxi
        self.passenger = passenger
        self.dest = dest
        self.v_s = v_s
        self.q_down = q_down
        self.q_up = q_up
        self.q_right = q_right
        self.q_left = q_left
        self.q_pickup = q_pickup
        self.q_dropoff = q_dropoff
        self.best_action = best_action
        self.epsilon = epsilon

    def updateVs(self):
    #the updateVs has to come after you updated the value of every other TaxiStateValueRow
    #because the first part is finding all the Q(s,a), second part will be updating V(S) and epsilon
    #if you update V(S) first, the other Q(s,a) of other TaxiStateValueRow will be wrong, because they will be taking from this V(S) that is updated too early
        ACTION_NAMES = ['down', 'up', 'right', 'left', 'pickup', 'dropoff']
        q_values = [self.q_down, self.q_up, self.q_right, self.q_left, self.q_pickup, self.q_dropoff]
        best_index = np.argmax(q_values)
        self.best_action = ACTION_NAMES[best_index]
        MaxQValue =max(q_values)
        self.epsilon = abs(MaxQValue - self.v_s)
        self.v_s = max(self.q_down, self.q_up, self.q_right, self.q_left, self.q_pickup, self.q_dropoff)

    def calculateQValues(self,VTable:dict[Tuple[int,int,int,int],Self]):
        for action in range(6):
            env.reset()
            env.unwrapped.s = encode_state(self.x_taxi, self.y_taxi, self.passenger, self.dest) # Manually set state
            newState, reward, done, truncated, info = env.step(action)
            newStatee = decode_state(newState)
            Q_S_A = reward + self.discount * (VTable[newStatee].v_s if not done else 0)#if terminal state, V(S') is 0

            # print(f'action is {action_names[action]}, current state is {(self.x_taxi, self.y_taxi, self.passenger, self.dest)}')
            # print(f'new state is {newStatee}, reward: {reward}, done: {done}, truncated: {truncated}, info: {info}')
            # print(f'state is {(self.x_taxi, self.y_taxi, self.passenger, self.dest)}, action is {action_names[action]}, Q(S,A) is {Q_S_A}')
            # print(newStatee, Q_S_A)
            # print("===============================================")
            if action == 0:
                self.q_down = Q_S_A
            elif action == 1:
                self.q_up = Q_S_A
            elif action == 2:
                self.q_right = Q_S_A
            elif action == 3:
                self.q_left = Q_S_A
            elif action == 4:
                self.q_pickup = Q_S_A
            elif action == 5:
                self.q_dropoff = Q_S_A
        #print('NEW QSA',self.q_down, self.q_up, self.q_right, self.q_left, self.q_pickup, self.q_dropoff)  # name (e.g. 'right')
        #print(f'v_s: {self.v_s}, epsilon: {self.epsilon}')

    def to_dict(self):
        return {
            'x_taxi': self.x_taxi,
            'y_taxi': self.y_taxi,
            'Passenger': self.passenger,
            'Dest': self.dest,
            'V(S)': self.v_s,
            'Q(down)': self.q_down,
            'Q(up)': self.q_up,
            'Q(right)': self.q_right,
            'Q(left)': self.q_left,
            'Q(pickup)': self.q_pickup,
            'Q(dropoff)': self.q_dropoff,
            'Best action': self.best_action,
            'epsilon': self.epsilon,
        }
    def __str__(self):
        #print out self as a string
        return ', '.join([f'{key}: {value}' for key, value in self.to_dict().items()])

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

#initialise state value  stable
x_range = range(5)         # 0 to 4
y_range = range(5)         # 0 to 4
passenger_values = [0,2,3, 4]  # we have to include all passenger locations, even though we only want pasenger at 1, because dropping off passenger at wrong hotel is allowed to change state, although get reward of -10
dest_values = [1]          # only 1

combinations = list(product(x_range, y_range, passenger_values, dest_values))
VTable:dict[Tuple[int,int,int,int],TaxiStateValueRow] ={}#tuple : TaxiStateValueRow

for x_taxi, y_taxi, passenger, dest in combinations:
  # print(f"(x_taxi, y_taxi, passenger, dest) = ({x_taxi}, {y_taxi}, {passenger}, {dest})")
   VTable[(x_taxi, y_taxi, passenger, dest)] = TaxiStateValueRow(x_taxi, y_taxi, passenger, dest,0,0,0,0,0,0,0,None,0)

# while epsilon<maxEpsilon:
#     epsilonVal = []
#     #calculate Q value of every state and action pair
#     for key, value in VTable.items():
#         value.calculateQValues(VTable)
#         #print(value)

#     #finished calculation, time to update Vtable
#     for key, value in VTable.items():
#         value.updateVs()
#         epsilonVal.append(value.epsilon)

#     maxEpsilon= max(epsilonVal)

#     print(f'Finished updating whole Vtable, max epsilon ={maxEpsilon}==================================\n ',i)
#     i+=1


# Convert dictionary of TaxiStateValueRow to DataFrame
# data = [row.to_dict() for row in VTable.values()]

# df = pd.DataFrame(data)
optimalAction = [
    (0,0,0,1),
    (0,0,4,1),
    (0,1,4,1),
    (0,2,4,1),
    (1,2,4,1),
    (2,2,4,1),
    (2,1,4,1),
    (2,0,4,1),
    (3,0,4,1),
    (4,0,4,1),
]
# df_sorted = df.sort_values(by=['Passenger', 'x_taxi', 'y_taxi', 'Dest'])
# # Sort by passenger, then x_taxi, then y_taxi, then dest
# states.sort(key=lambda s: (s.passenger, s.x_taxi, s.y_taxi, s.dest))
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

class LivePandasTable:
    def __init__(self, root, initial_df: pd.DataFrame):
        self.root = root
        self.root.title("Taxi Value Table - Auto Refresh Every 0.2s")

        self.tree = ttk.Treeview(root, columns=list(initial_df.columns), show='headings')
        for col in initial_df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=90)
        self.tree.pack(expand=True, fill='both')

        self.current_df = initial_df.copy()
        self.update_table(self.current_df)

    def update_table(self, df):
        self.tree.delete(*self.tree.get_children())
        for _, row in df.iterrows():
            self.tree.insert('', 'end', values=list(row))

    def refresh(self, new_df: pd.DataFrame):
        self.current_df = new_df.copy()
        self.update_table(self.current_df)

# Initialize DataFrame from VTable objects (your actual VTable logic here)
data = [row.to_dict() for row in VTable.values()]
df = pd.DataFrame(data)
df_sorted = df.sort_values(by=['Passenger', 'x_taxi', 'y_taxi', 'Dest'])

root = tk.Tk()
app = LivePandasTable(root, df_sorted)

iteration = 1
converged = False

def auto_update():
    global iteration, converged

    if converged:
        return  # stop updates once converged

    epsilonVals = []

    # Recalculate values
    for key, value in VTable.items():
        value.calculateQValues(VTable)
    for key, value in VTable.items():
        value.updateVs()
        epsilonVals.append(value.epsilon)

    maxEpsilon = max(epsilonVals)
    print(f'iteration: {iteration}, Max Epsilon: {maxEpsilon}')

    # Update DataFrame and refresh UI
    updated_data = [row.to_dict() for row in VTable.values()]
    filtered_data = [
        row for row in updated_data
        if (row['x_taxi'], row['y_taxi'], row['Passenger'], row['Dest']) in optimalAction
    ]
    updated_df = pd.DataFrame(filtered_data).sort_values(by=['Passenger', 'x_taxi', 'y_taxi', 'Dest'])
    app.refresh(updated_df)

    if maxEpsilon < 1e-5 and not converged:
        messagebox.showinfo("Info", f"epsilon converged at iteration {iteration}")
        converged = True

    iteration += 1
    if not converged:
        root.after(200, auto_update)  # schedule next update in 0.2 seconds

# Start auto update loop after 200 ms
root.after(200, auto_update)

root.mainloop()
