import numpy as np
import gymnasium as gym
import tkinter as tk
from tkinter import ttk
from sklearn.neural_network import MLPRegressor
import threading
import random
import time

# === GUI Spreadsheet ===
class LiveQStepDisplay:
    def __init__(self, max_rows=50):
        self.root = tk.Tk()
        self.root.title("Live DQN Step Viewer")
        self.tree = ttk.Treeview(self.root)
        self.tree.pack(fill="both", expand=True)

        self.columns = ["Step", "Episode", "State", "Action", "Q-Values", "Max Q", "Reward",
                        "Next State", "TD Error", "Best Action", "Done", "Epsilon"]
        self.tree["columns"] = self.columns
        self.tree["show"] = "headings"
        for col in self.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER, width=120)

        self.data = []
        self.max_rows = max_rows

    def log_step(self, step, episode, state, action, q_vals, reward, next_state, td_error, done, epsilon):
        decode = lambda s: str(np.unravel_index(s, (5, 5, 5, 4)))
        action_map = ["down", "up", "right", "left", "pickup", "dropoff"]
        max_q = round(float(np.max(q_vals)), 3)
        q_str = "[" + ", ".join(f"{round(float(q), 2)}" for q in q_vals) + "]"
        best_action = action_map[np.argmax(q_vals)]

        row = [step, episode, decode(state), action_map[action], q_str, max_q,
               reward, decode(next_state), round(float(td_error), 4), best_action, done, round(epsilon, 3)]

        self.data.append(row)
        if len(self.data) > self.max_rows:
            self.data.pop(0)

    def refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for entry in self.data:
            self.tree.insert("", "end", values=entry)
        self.root.update()

    def run(self):
        self.root.mainloop()

# === MLP Visualizer ===
class MLPVisualizer:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("MLP Neural Network Map")
        self.canvas = tk.Canvas(self.root, width=900, height=600, bg="white")
        self.canvas.pack()

        self.neuron_radius = 15
        self.neuron_coords = []
        self.edge_lines = []
        self._draw_network()

    def _draw_network(self):
        layer_sizes = [self.model.coefs_[0].shape[0]] + [w.shape[1] for w in self.model.coefs_]
        x_spacing = self.canvas.winfo_reqwidth() // (len(layer_sizes) + 1)

        for i, layer_size in enumerate(layer_sizes):
            y_spacing = self.canvas.winfo_reqheight() // (layer_size + 1)
            layer = []
            for j in range(layer_size):
                x = (i + 1) * x_spacing
                y = (j + 1) * y_spacing
                neuron = self.canvas.create_oval(
                    x - self.neuron_radius, y - self.neuron_radius,
                    x + self.neuron_radius, y + self.neuron_radius,
                    fill="gray", outline="black"
                )
                layer.append((x, y, neuron))
            self.neuron_coords.append(layer)

        for l in range(len(self.neuron_coords) - 1):
            edge_layer = []
            for i, (x1, y1, _) in enumerate(self.neuron_coords[l]):
                for j, (x2, y2, _) in enumerate(self.neuron_coords[l + 1]):
                    line = self.canvas.create_line(x1, y1, x2, y2, fill="gray")
                    edge_layer.append(((i, j), line))
            self.edge_lines.append(edge_layer)

        self.canvas.update()

    def updateVisual(self, input_vector):
        activations = [input_vector.flatten()]
        for i in range(len(self.model.coefs_)):
            z = np.dot(activations[-1], self.model.coefs_[i]) + self.model.intercepts_[i]
            a = np.tanh(z)
            activations.append(a)

        for layer_index, layer in enumerate(self.neuron_coords):
            for i, (_, _, neuron_id) in enumerate(layer):
                if layer_index < len(activations) and i < len(activations[layer_index]):
                    act = abs(activations[layer_index][i])
                    red = int(min(255, 255 * act))
                    color = f'#{red:02x}88ff'
                    self.canvas.itemconfig(neuron_id, fill=color)

        for l, edge_layer in enumerate(self.edge_lines):
            weights = self.model.coefs_[l]
            for (i, j), line_id in edge_layer:
                if i < weights.shape[0] and j < weights.shape[1]:
                    w = abs(weights[i][j])
                    gray = int(min(255, w * 255))
                    color = f'#{gray:02x}{gray:02x}{gray:02x}'
                    self.canvas.itemconfig(line_id, fill=color)

        self.canvas.update_idletasks()
        self.canvas.update()

    def run(self):
        self.root.mainloop()

# === Feature Encoder ===
def encode_state_features(state):
    x, y, p, d = np.unravel_index(state, (5, 5, 5, 4))
    return np.array([x, y, p, d]) / np.array([4, 4, 4, 3])  # normalize

# === DQN Training with feature encoding ===
def run_dqn(display):
    env = gym.make("Taxi-v3")
    state_space = env.observation_space.n
    action_space = env.action_space.n
    epsilon = 0.5
    alpha = 0.1
    gamma = 0.99
    episodes = 200
    step_counter = 0
    model = MLPRegressor(hidden_layer_sizes=(16, 12), max_iter=1, warm_start=True)
    model.partial_fit(np.zeros((1, 4)), np.zeros((1, action_space)))  # initialize with 4D input

    X, y = [], []

    for episode in range(1, episodes + 1):
        state = env.reset()[0]
        done = False

        while not done:
            input_vec = encode_state_features(state)
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, action_space - 1)
            else:
                action = np.argmax(model.predict(input_vec.reshape(1, -1))[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            q_vals_current = model.predict(input_vec.reshape(1, -1))[0]
            q_vals_next = model.predict(encode_state_features(next_state).reshape(1, -1))[0]
            target = q_vals_current.copy()
            td_target = reward + gamma * np.max(q_vals_next)
            td_error = td_target - q_vals_current[action]
            target[action] += alpha * td_error

            X.append(input_vec)
            y.append(target)

            if len(X) > 64:
                model.partial_fit(np.array(X[-64:]), np.array(y[-64:]))

            display.log_step(step_counter, episode, state, action, q_vals_current,
                             reward, next_state, td_error, done, epsilon)
            step_counter += 1

            if step_counter % 10 == 0:
                display.refresh()

            state = next_state

# === Mode 1: Spreadsheet + DQN only ===
def run_spreadsheet_only():
    display = LiveQStepDisplay()
    threading.Thread(target=run_dqn, args=(display,), daemon=True).start()
    display.run()

# === Mode 2: Visualizer map only ===
def run_visualizer_only():
    dummy_model = MLPRegressor(hidden_layer_sizes=(16, 12))
    dummy_model.partial_fit(np.zeros((1, 4)), np.zeros((1, 6)))  # 4 inputs, 6 actions

    visualizer = MLPVisualizer(dummy_model)
    sample_input = np.array([[0, 0, 0, 0]])  # Normalized [x, y, passenger, destination]
    visualizer.updateVisual(sample_input)
    visualizer.run()

# === Select Mode ===
if __name__ == "__main__":
    #==================================================
    # just comment out accordingly, if not is very laggy

    #run_spreadsheet_only()
    run_visualizer_only()
