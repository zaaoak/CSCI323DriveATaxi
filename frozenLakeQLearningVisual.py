import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
import time

# for tkinter visualizetion
cell_size = 50
arrow_symbols = ['←', '↓', '→', '↑']  # Action indicators

root = tk.Tk()
root.title("Live Q-Table Visualization")

frame_border = 10
canvas = tk.Canvas(root, width=4 * cell_size + 2 * frame_border,
                   height=4 * cell_size + 2 * frame_border + 30, bg="white", highlightthickness=1, highlightbackground="black")
canvas.pack(padx=frame_border, pady=frame_border)

episode_text = canvas.create_text(4 * cell_size // 2 + frame_border, 4 * cell_size + 20,
                                   text="Episode: 0", font=("Arial", 12))

cells = [[None for _ in range(4)] for _ in range(4)]
texts = [[None for _ in range(4)] for _ in range(4)]

for i in range(4):
    for j in range(4):
        x0 = j * cell_size + frame_border
        y0 = i * cell_size + frame_border
        rect = canvas.create_rectangle(x0, y0, x0 + cell_size, y0 + cell_size, fill="white")
        text = canvas.create_text(x0 + cell_size // 2, y0 + cell_size // 2, text="", font=("Arial", 12))
        cells[i][j] = rect
        texts[i][j] = text

plt.ion()
fig, ax = plt.subplots(figsize=(6, 4))
line, = ax.plot([], [])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Episode")
ax.set_ylabel("Average reward")
ax.set_title("Rolling Average of Episode Rewards")
fig.tight_layout()

def update_q_visual(q, current_episode, reward_plot):
    max_q = np.max(q)
    for state in range(16):
        i, j = divmod(state, 4)
        best_action = np.argmax(q[state])
        value = np.max(q[state])
        intensity = int(255 * value / max_q) if max_q > 0 else 0
        color = f'#{"%02x"%(255 - intensity)}{"%02x"%(255 - intensity)}ff'
        canvas.itemconfig(cells[i][j], fill=color)
        canvas.itemconfig(texts[i][j], text=f"{arrow_symbols[best_action]}\n{value:.2f}")
    canvas.itemconfig(episode_text, text=f"Episode: {current_episode}")
    root.update()

    line.set_data(np.arange(len(reward_plot)), reward_plot)
    ax.set_xlim(0, len(reward_plot))
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    fig.canvas.flush_events()

def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode='human' if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array

    learning_rate_a = 0.01 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.001       # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    rolling_avg_rewards = []

    window_size = 100

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

        start = max(0, i - window_size + 1)
        rolling_avg = np.mean(rewards_per_episode[start:i + 1])
        rolling_avg_rewards.append(rolling_avg)

        update_q_visual(q, i + 1, rolling_avg_rewards)

    env.close()

    plt.ioff()

    fig, ax2 = plt.subplots()
    q_max = np.max(q, axis=1).reshape((4, 4))
    best_actions = np.argmax(q, axis=1).reshape((4, 4))
    im = ax2.imshow(q_max, cmap="Blues", vmin=0.0, vmax=1.0)

    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f"{arrow_symbols[best_actions[i, j]]}\n{q_max[i, j]:.2f}",
                     ha='center', va='center', color='black', fontsize=10)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Learned Q-values\nArrows represent best action", pad=15)
    ax2.set_frame_on(True)
    fig.colorbar(im, ax=ax2, shrink=0.8, label="Q-value")
    plt.tight_layout(pad=2)
    plt.show()

if __name__ == '__main__':
    run(3000, is_training=True, render=False)
    root.mainloop()
