from collections import defaultdict
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple
from gymnasium.wrappers import TimeLimit
import tkinter as tk
from itertools import product
import threading
import time
# Setup action names and decoding function
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
    return taxi_col, taxi_row, passenger, dest

def encode_state(taxi_col, taxi_row, passenger_loc, destination):
    return ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination

# Initialize environment
env = gym.make('Taxi-v3', render_mode="ansi")
env = TimeLimit(env.unwrapped, max_episode_steps=200)

# Setup initial state
passengerStart = 0
Destination = 1
env.reset()
initial_state = encode_state(0, 0, passengerStart, Destination)
env.unwrapped.s = initial_state

# RL Parameters
epsilon = 1e-5
maxEpsilon = 1
discount = 0.4

# Create VTable and PolicyTable
x_range = range(5)
y_range = range(5)
passenger_values = [0, 2, 3, 4]
dest_values = [1]
combinations = list(product(x_range, y_range, passenger_values, dest_values))

VTable: dict[Tuple[int, int, int, int], float] = {item: 0.0 for item in combinations}
PolicyTable = {item: 0 for item in combinations}

# Tkinter GUI setup
root = tk.Tk()
root.title("Live VTable and PolicyTable Display")

text_widget = tk.Text(root, wrap='none', font=("Courier", 9))
text_widget.pack(expand=True, fill='both')
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
def update_display():
    filtered_states = {k: v for k, v in VTable.items() if k in optimalAction}
    filtered_policy = {k: v for k, v in PolicyTable.items() if k in optimalAction}
    df = pd.DataFrame({
        'State': list(filtered_states.keys()),
        'V(s)': [round(filtered_states[s], 2) for s in filtered_states],
        'Policy (Action)': [action_names[filtered_policy[s]] for s in filtered_policy]
    })
    text_widget.delete('1.0', tk.END)
    text_widget.insert(tk.END, df.to_string(index=False))
    root.after(200, update_display)

def run_policy_iteration():
    global VTable, PolicyTable
    converged = False
    iterations = 0
    steps = 0
    while not converged:
        iterations += 1
        epsilonThreshhold = 1e-5
        while epsilon < maxEpsilon:
            epsilonTab = []
            for x_taxi, y_taxi, passenger, dest in combinations:
                env.reset()
                env.unwrapped.s = encode_state(x_taxi, y_taxi, passenger, dest)
                actionToTake = PolicyTable[(x_taxi, y_taxi, passenger, dest)]
                newState, reward, done, truncated, info = env.step(actionToTake)
                futureV = 0 if done else VTable[decode_state(newState)]
                v = reward + discount * futureV
                epsilonTab.append(abs(VTable[(x_taxi, y_taxi, passenger, dest)] - v))
                VTable[(x_taxi, y_taxi, passenger, dest)] = v
            max_epsilon = max(epsilonTab)
            steps += 1
            time.sleep(0.2)  # Add this line to allow GUI update pacing
            if max_epsilon < epsilonThreshhold:
                print(f"Policy evaluation converged after {steps} steps. moving onto policy improvement")
                break
        converged = True
        for x_taxi, y_taxi, passenger, dest in combinations:
            QValues = [0.0] * 6
            for action in range(6):
                env.reset()
                env.unwrapped.s = encode_state(x_taxi, y_taxi, passenger, dest)
                newState, reward, done, truncated, info = env.step(action)
                futureV = 0 if done else VTable[decode_state(newState)]
                QValues[action] = reward + discount * futureV
            best_action = QValues.index(max(QValues))
            current_action = PolicyTable[(x_taxi, y_taxi, passenger, dest)]
            if current_action != best_action:
                converged = False
                PolicyTable[(x_taxi, y_taxi, passenger, dest)] = best_action
    print(f"Policy iteration converged after {iterations} iterations and {steps} steps.")

# Start display updates
update_display()
text_widget.tag_configure("highlight", background="yellow", foreground="black")

# Start training in a separate thread
training_thread = threading.Thread(target=run_policy_iteration)
training_thread.daemon = True
training_thread.start()

# Run GUI event loop
root.mainloop()
