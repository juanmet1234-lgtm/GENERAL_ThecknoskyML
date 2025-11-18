import os
import io
import random
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

# --------------------------------------------------
# GridWorld
# --------------------------------------------------
class GridWorld:
    def __init__(self, size=4):
        self.size = int(size)
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.state = self.start

        self.n_rows = self.size
        self.n_cols = self.size
        self.grid_shape = (self.size, self.size)
        self.n_states = self.size * self.size
        self.n_actions = 4
        self.start_state = self.get_state_id(self.start)
        self.goal_state = self.get_state_id(self.goal)

    def reset(self):
        self.state = self.start
        return self.get_state_id(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.size - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.size - 1, y + 1)

        self.state = (x, y)

        reward = -1
        done = False
        if self.state == self.goal:
            reward = 10
            done = True

        return self.get_state_id(self.state), reward, done

    def get_state_id(self, state):
        return int(state[0]) * self.size + int(state[1])

    def step_from_state(self, state_id, action):
        r = int(state_id) // self.size
        c = int(state_id) % self.size
        x, y = r, c

        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.size - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.size - 1, y + 1)

        next_state = (x, y)
        next_id = self.get_state_id(next_state)

        reward = -1
        done = False
        if next_state == self.goal:
            reward = 10
            done = True

        return next_id, reward, done


# --------------------------------------------------
# Q-Learning Agent
# --------------------------------------------------
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1,
                 epsilon_decay=0.995, epsilon_min=0.01):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.n_states = env.n_states
        self.n_actions = env.n_actions

        self.q_table = np.zeros((self.n_states, self.n_actions))

    def choose_action(self, state_id, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[state_id]))

    def train(self, episodes=200, max_steps=100):
        rewards_history = []
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                best_next = np.max(self.q_table[next_state])
                td = reward + self.gamma * best_next - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td

                state = next_state
                total_reward += reward
                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)

        return rewards_history

    def save_qtable(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.q_table)

    def load_qtable(self, filepath):
        self.q_table = np.load(filepath)

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def simulate_policy(self, env=None, start_state=None, max_steps=100):
        env = env or self.env
        state = start_state if start_state is not None else env.start_state

        traj = [state]
        policy = self.get_policy()

        for _ in range(max_steps):
            action = int(policy[state])
            next_state, reward, done = env.step_from_state(state, action)
            traj.append(next_state)
            state = next_state
            if done:
                break

        return traj


# --------------------------------------------------
# VISUALIZACIÓN
# --------------------------------------------------
def plot_policy(env, agent, out_path):
    policy = agent.get_policy()

    if len(policy) != env.n_states:
        raise ValueError("Q-table y entorno no coinciden en tamaño.")

    grid = policy.reshape(env.grid_shape)

    plt.figure(figsize=(env.size, env.size))
    plt.imshow(grid, cmap='tab10')
    plt.title("Policy (argmax Q)")
    plt.xticks([])
    plt.yticks([])

    gr, gc = env.goal
    plt.text(gc, gr, "G", ha="center", va="center", color="white", fontsize=14)

    sr, sc = env.start
    plt.text(sc, sr, "S", ha="center", va="center", color="white", fontsize=14)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_rewards(rewards, window=10):
    plt.figure(figsize=(6, 3))
    plt.plot(rewards)

    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(smooth)

    plt.title("Recompensa por episodio")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    encoded = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return encoded


def plot_trajectory(traj, size=4):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.invert_yaxis()

    xs = [t % size for t in traj]
    ys = [t // size for t in traj]

    ax.plot(xs, ys, marker="o")
    ax.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    encoded = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return encoded


def create_trajectory_gif(env, traj, out_path, fps=4):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    frames = []
    size = env.size

    for state in traj:
        fig, ax = plt.subplots(figsize=(size, size))
        ax.invert_yaxis()

        xs = [state % size]
        ys = [state // size]

        ax.scatter(xs, ys, s=300, c="red")
        ax.grid()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        frames.append(imageio.v2.imread(buf))

        buf.close()
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path
