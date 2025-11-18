import os
import io
import random
import base64
import numpy as np
import matplotlib
# Usar backend no interactivo para servidores / Flask (evita uso de Tk)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from matplotlib import colors

# --------------------------------------------------
# Entorno simple tipo GridWorld
# --------------------------------------------------
class GridWorld:
    def __init__(self, size=4):
        self.size = int(size)
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.state = self.start

        # atributos útiles para el agente / utilidades
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
        if action == 0:   # up
            x = max(0, x - 1)
        elif action == 1: # down
            x = min(self.size - 1, x + 1)
        elif action == 2: # left
            y = max(0, y - 1)
        elif action == 3: # right
            y = min(self.size - 1, y + 1)

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
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.size - 1, y + 1)

        next_state = (x, y)
        next_id = self.get_state_id(next_state)
        reward = -1
        done = False
        if next_state == self.goal:
            reward = 10
            done = True
        return next_id, reward, done

# --------------------------------------------------
# Agente Q-Learning
# --------------------------------------------------
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # inferir n_states / n_actions desde el entorno
        n_states = getattr(env, "n_states", None)
        if n_states is None:
            if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
                n_states = int(env.n_rows) * int(env.n_cols)
        n_actions = getattr(env, "n_actions", None)
        if n_actions is None:
            n_actions = getattr(env, "nA", None)

        if n_states is None or n_actions is None:
            raise AttributeError("Proporcione atributos n_states/n_actions (o n_rows/n_cols).")

        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def choose_action(self, state_id, greedy=False):
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[int(state_id)]))

    def train(self, episodes=200, max_steps=100):
        rewards_history = []
        for ep in range(int(episodes)):
            state = self.env.reset()
            total_reward = 0
            for _ in range(int(max_steps)):
                action = self.choose_action(state, greedy=False)
                next_state, reward, done = self.env.step(action)
                best_next = np.max(self.q_table[next_state])
                td = reward + self.gamma * best_next - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td
                state = next_state
                total_reward += reward
                if done:
                    break
            # decay epsilon
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
        state = start_state if start_state is not None else getattr(env, "start_state", env.reset())
        traj = [int(state)]
        policy = self.get_policy()
        for _ in range(int(max_steps)):
            action = int(policy[int(state)])
            if hasattr(env, "step_from_state"):
                next_state, reward, done = env.step_from_state(int(state), int(action))
            else:
                # best-effort: set internal state and call step()
                try:
                    r = int(state) // getattr(env, "n_cols", int(np.sqrt(self.n_states)))
                    c = int(state) % getattr(env, "n_cols", int(np.sqrt(self.n_states)))
                    env.state = (r, c)
                except Exception:
                    pass
                next_state, reward, done = env.step(action)
            traj.append(int(next_state))
            state = next_state
            if done:
                break
        return traj

# --------------------------------------------------
# Visualización / utilidades
# --------------------------------------------------
def plot_policy(env, agent, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    policy = agent.get_policy()
    try:
        rows, cols = env.grid_shape
        policy_grid = policy.reshape((rows, cols))
    except Exception:
        try:
            rows, cols = env.n_rows, env.n_cols
            policy_grid = policy.reshape((rows, cols))
        except Exception:
            policy_grid = policy[np.newaxis, :]
            rows, cols = policy_grid.shape

    cmap = plt.cm.get_cmap('tab10', int(np.max(policy_grid) - np.min(policy_grid) + 1))
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.imshow(policy_grid, cmap=cmap, interpolation='nearest')
    ax.set_title("Policy (argmax Q)")
    ax.set_xticks([])
    ax.set_yticks([])

    if hasattr(env, "goal_state"):
        try:
            g_r, g_c = divmod(int(env.goal_state), policy_grid.shape[1])
            ax.text(g_c, g_r, "G", ha="center", va="center", color="white", fontsize=12, weight="bold")
        except Exception:
            pass
    if hasattr(env, "start_state"):
        try:
            s_r, s_c = divmod(int(env.start_state), policy_grid.shape[1])
            ax.text(s_c, s_r, "S", ha="center", va="center", color="white", fontsize=12, weight="bold")
        except Exception:
            pass

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def create_trajectory_gif(env, traj, out_path, fps=4):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        rows, cols = env.grid_shape
    except Exception:
        try:
            rows, cols = env.n_rows, env.n_cols
        except Exception:
            n = getattr(env, "n_states", len(traj))
            side = int(np.ceil(np.sqrt(n)))
            rows, cols = side, side

    frames = []
    for state in traj:
        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xticks([])
        ax.set_yticks([])
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, edgecolor='lightgray', facecolor='white'))
        if hasattr(env, "goal_state"):
            gr, gc = divmod(int(env.goal_state), cols)
            ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1, color='green', alpha=0.4))
            ax.text(gc, gr, "G", ha="center", va="center", color="white")
        if hasattr(env, "start_state"):
            sr, sc = divmod(int(env.start_state), cols)
            ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color='blue', alpha=0.4))
            ax.text(sc, sr, "S", ha="center", va="center", color="white")
        ar, ac = divmod(int(state), cols)
        ax.plot(ac, ar, 'o', color='red', markersize=16)
        ax.set_xlim(-0.6, cols - 0.4)
        ax.set_ylim(rows - 0.4, -0.6)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = imageio.v2.imread(buf)
        frames.append(image)
        buf.close()
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path

def plot_rewards(rewards, window=20):
    plt.figure(figsize=(6, 3))
    plt.plot(rewards, label="Recompensa por episodio")
    if len(rewards) >= 2:
        try:
            smoothed = np.convolve(rewards, np.ones(min(window, len(rewards)))/min(window, len(rewards)), mode='valid')
            plt.plot(range(len(smoothed)), smoothed, label="Media móvil")
        except Exception:
            pass
    plt.title("Recompensa por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded

def plot_trajectory(traj, size=4):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_xlim(-0.5, size-0.5)
    ax.set_ylim(-0.5, size-0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.invert_yaxis()
    ax.grid(True)
    xs = [p % size for p in traj]
    ys = [p // size for p in traj]
    ax.plot(xs, ys, marker='o', color='C1')
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded
