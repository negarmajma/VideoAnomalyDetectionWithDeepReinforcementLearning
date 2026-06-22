#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# Section 0: Create Model Directory
# ============================================================

models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

print(models_dir)


# In[2]:


# ============================================================
# Section 1: Import Required Libraries
# ============================================================

from pathlib import Path
import numpy as np


# In[3]:


# ============================================================
# Section 2: Load Anomaly Scores and Test Labels
# ============================================================

project_root = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue")
dataset_dir = project_root / "rl_dataset"

anomaly_scores = np.load(dataset_dir / "anomaly_scores.npy")
y_test = np.load(dataset_dir / "y_test.npy")

print("Anomaly scores:", anomaly_scores.shape)
print("Labels:", y_test.shape)
print("Normal:", np.sum(y_test == 0))
print("Anomaly:", np.sum(y_test == 1))


# In[4]:


# ============================================================
# Section 3: Normalize Anomaly Scores
# ============================================================

score_min = anomaly_scores.min()
score_max = anomaly_scores.max()

scores_norm = (anomaly_scores - score_min) / (score_max - score_min)

print("Normalized min:", scores_norm.min())
print("Normalized max:", scores_norm.max())


# In[5]:


# ============================================================
# Section 4: Build RL State Features
# ============================================================

states = []

for i in range(len(scores_norm)):

    current_score = scores_norm[i]

    if i == 0:
        previous_score = scores_norm[i]
    else:
        previous_score = scores_norm[i - 1]

    score_change = current_score - previous_score

    state = [
        current_score,
        previous_score,
        score_change
    ]

    states.append(state)

states = np.array(states, dtype=np.float32)

print("States shape:", states.shape)
print("First state:", states[0])


# In[6]:


# ============================================================
# Section 5: Define Video Anomaly RL Environment
# ============================================================

class VideoAnomalyEnv:

    def __init__(self, states, labels):

        self.states = states
        self.labels = labels
        self.current_index = 0
        self.n_samples = len(states)

    def reset(self):

        self.current_index = 0
        return self.states[self.current_index]

    def step(self, action):

        true_label = self.labels[self.current_index]

        # Reward design
        if action == 1 and true_label == 1:
            reward = 10      # True Positive
        elif action == 0 and true_label == 0:
            reward = 2       # True Negative
        elif action == 1 and true_label == 0:
            reward = -3      # False Positive
        elif action == 0 and true_label == 1:
            reward = -10     # False Negative

        self.current_index += 1

        done = self.current_index >= self.n_samples

        if done:
            next_state = np.zeros_like(self.states[0])
        else:
            next_state = self.states[self.current_index]

        return next_state, reward, done, true_label


# In[7]:


# ============================================================
# Section 6: Test RL Environment
# ============================================================

env = VideoAnomalyEnv(states, y_test)

state = env.reset()

print("Initial state:", state)

next_state, reward, done, true_label = env.step(action=0)

print("Next state:", next_state)
print("Reward:", reward)
print("Done:", done)
print("True label:", true_label)


# In[8]:


# ============================================================
# Section 7: Import Deep Learning Libraries for DQN
# ============================================================

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# In[9]:


# ============================================================
# Section 8: Configure Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


# In[10]:


# ============================================================
# Section 9: Define DQN Network
# ============================================================

class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, action_size)
        )

    def forward(self, x):
        return self.network(x)


# In[11]:


# ============================================================
# Section 10: Initialize DQN Parameters
# ============================================================

state_size = 3
action_size = 2

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

criterion = nn.MSELoss()

print(policy_net)


# In[12]:


# ============================================================
# Section 11: Define Replay Memory and Action Selection
# ============================================================

memory = deque(maxlen=5000)

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

batch_size = 32
target_update_freq = 10


def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return torch.argmax(q_values).item()


# In[13]:


# ============================================================
# Section 12: Define DQN Training Step
# ============================================================

def train_dqn_step():

    if len(memory) < batch_size:
        return None

    batch = random.sample(memory, batch_size)

    states_batch = torch.tensor(
        np.array([item[0] for item in batch]),
        dtype=torch.float32
    ).to(device)

    actions_batch = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long
    ).unsqueeze(1).to(device)

    rewards_batch = torch.tensor(
        [item[2] for item in batch],
        dtype=torch.float32
    ).unsqueeze(1).to(device)

    next_states_batch = torch.tensor(
        np.array([item[3] for item in batch]),
        dtype=torch.float32
    ).to(device)

    dones_batch = torch.tensor(
        [item[4] for item in batch],
        dtype=torch.float32
    ).unsqueeze(1).to(device)

    q_values = policy_net(states_batch).gather(
        1,
        actions_batch
    )

    with torch.no_grad():
        next_q_values = target_net(next_states_batch).max(1)[0].unsqueeze(1)
        target_q_values = rewards_batch + gamma * next_q_values * (1 - dones_batch)

    loss = criterion(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# In[14]:


# ============================================================
# Section 13: Train DQN Agent
# ============================================================

num_episodes = 100

episode_rewards = []
episode_losses = []

for episode in range(num_episodes):

    state = env.reset()
    total_reward = 0
    losses = []
    done = False

    while not done:

        action = select_action(state, epsilon)

        next_state, reward, done, true_label = env.step(action)

        memory.append(
            (state, action, reward, next_state, done)
        )

        loss = train_dqn_step()

        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    avg_loss = np.mean(losses) if losses else 0

    episode_rewards.append(total_reward)
    episode_losses.append(avg_loss)

    print(
        f"Episode {episode+1}/{num_episodes} | "
        f"Reward: {total_reward:.2f} | "
        f"Loss: {avg_loss:.4f} | "
        f"Epsilon: {epsilon:.4f}"
    )


# In[17]:


# ============================================================
# Save Figure 4-3: DQN Episode Reward with Moving Average
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

rewards = np.array(episode_rewards)

window_size = 10
moving_avg = np.convolve(
    rewards,
    np.ones(window_size) / window_size,
    mode="valid"
)

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(rewards) + 1),
    rewards,
    alpha=0.4,
    label="Episode Reward"
)

plt.plot(
    range(window_size, len(rewards) + 1),
    moving_avg,
    linewidth=2.5,
    label="Moving Average (10 Episodes)"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend(fontsize=10)

save_path = results_dir / "figure_4_3_dqn_reward_moving_average.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)


# In[14]:


# ============================================================
# Section 14: Plot DQN Training Reward
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Reward")
plt.grid(True)
plt.show()


# In[15]:


# ============================================================
# Section 15: Evaluate Trained DQN Agent
# ============================================================

env_eval = VideoAnomalyEnv(states, y_test)

state = env_eval.reset()
done = False

dqn_predictions = []
true_labels = []

while not done:

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    action = torch.argmax(q_values).item()

    next_state, reward, done, true_label = env_eval.step(action)

    dqn_predictions.append(action)
    true_labels.append(true_label)

    state = next_state

dqn_predictions = np.array(dqn_predictions)
true_labels = np.array(true_labels)

print("Predicted normal:", np.sum(dqn_predictions == 0))
print("Predicted anomaly:", np.sum(dqn_predictions == 1))


# In[16]:


# ============================================================
# Section 16: Evaluate DQN Performance
# ============================================================

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

acc = accuracy_score(true_labels, dqn_predictions)
pre = precision_score(true_labels, dqn_predictions, zero_division=0)
rec = recall_score(true_labels, dqn_predictions, zero_division=0)
f1 = f1_score(true_labels, dqn_predictions, zero_division=0)
cm = confusion_matrix(true_labels, dqn_predictions)

print("Accuracy :", acc)
print("Precision:", pre)
print("Recall   :", rec)
print("F1-score :", f1)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(true_labels, dqn_predictions, zero_division=0))


# In[17]:


#Reward Version 2.


# In[18]:


# ============================================================
# Section 17: Define RL Environment with Reward Version 2
# ============================================================

class VideoAnomalyEnvV2:

    def __init__(self, states, labels):

        self.states = states
        self.labels = labels
        self.current_index = 0
        self.n_samples = len(states)

    def reset(self):

        self.current_index = 0
        return self.states[self.current_index]

    def step(self, action):

        true_label = self.labels[self.current_index]

        # Reward Version 2
        if action == 1 and true_label == 1:
            reward = 10      # True Positive
        elif action == 0 and true_label == 0:
            reward = 5       # True Negative
        elif action == 1 and true_label == 0:
            reward = -6      # False Positive
        elif action == 0 and true_label == 1:
            reward = -10     # False Negative

        self.current_index += 1

        done = self.current_index >= self.n_samples

        if done:
            next_state = np.zeros_like(self.states[0])
        else:
            next_state = self.states[self.current_index]

        return next_state, reward, done, true_label


# In[19]:


# ============================================================
# Section 18: Reinitialize DQN for Reward Version 2
# ============================================================

policy_net_v2 = DQN(state_size, action_size).to(device)
target_net_v2 = DQN(state_size, action_size).to(device)

target_net_v2.load_state_dict(policy_net_v2.state_dict())
target_net_v2.eval()

optimizer_v2 = optim.Adam(policy_net_v2.parameters(), lr=0.001)

memory_v2 = deque(maxlen=5000)

epsilon_v2 = 1.0

print(policy_net_v2)


# In[21]:


# ============================================================
# Section 19: Define Action Selection Function for DQN V2
# ============================================================

def select_action_v2(state, epsilon_v2):

    if random.random() < epsilon_v2:
        return random.randint(0, action_size - 1)

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net_v2(state_tensor)

    return torch.argmax(q_values).item()


# In[22]:


# ============================================================
# Section 20: Define Training Step for DQN V2
# ============================================================

def train_dqn_step_v2():

    if len(memory_v2) < batch_size:
        return None

    batch = random.sample(memory_v2, batch_size)

    states_batch = torch.tensor(
        np.array([item[0] for item in batch]),
        dtype=torch.float32
    ).to(device)

    actions_batch = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long
    ).unsqueeze(1).to(device)

    rewards_batch = torch.tensor(
        [item[2] for item in batch],
        dtype=torch.float32
    ).unsqueeze(1).to(device)

    next_states_batch = torch.tensor(
        np.array([item[3] for item in batch]),
        dtype=torch.float32
    ).to(device)

    dones_batch = torch.tensor(
        [item[4] for item in batch],
        dtype=torch.float32
    ).unsqueeze(1).to(device)

    q_values = policy_net_v2(states_batch).gather(
        1,
        actions_batch
    )

    with torch.no_grad():
        next_q_values = target_net_v2(next_states_batch).max(1)[0].unsqueeze(1)

        target_q_values = rewards_batch + gamma * next_q_values * (1 - dones_batch)

    loss = criterion(q_values, target_q_values)

    optimizer_v2.zero_grad()
    loss.backward()
    optimizer_v2.step()

    return loss.item()


# In[23]:


# ============================================================
# Section 21: Train DQN Agent with Reward Version 2
# ============================================================

env_v2 = VideoAnomalyEnvV2(states, y_test)

num_episodes_v2 = 100

episode_rewards_v2 = []
episode_losses_v2 = []

for episode in range(num_episodes_v2):

    state = env_v2.reset()
    total_reward = 0
    losses = []
    done = False

    while not done:

        action = select_action_v2(state, epsilon_v2)

        next_state, reward, done, true_label = env_v2.step(action)

        memory_v2.append(
            (state, action, reward, next_state, done)
        )

        loss = train_dqn_step_v2()

        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward

    if episode % target_update_freq == 0:
        target_net_v2.load_state_dict(policy_net_v2.state_dict())

    epsilon_v2 = max(
        epsilon_min,
        epsilon_v2 * epsilon_decay
    )

    avg_loss = np.mean(losses) if losses else 0

    episode_rewards_v2.append(total_reward)
    episode_losses_v2.append(avg_loss)

    print(
        f"Episode {episode+1}/{num_episodes_v2} | "
        f"Reward: {total_reward:.2f} | "
        f"Loss: {avg_loss:.4f} | "
        f"Epsilon: {epsilon_v2:.4f}"
    )


# In[24]:


# ============================================================
# Section 22: Evaluate DQN Agent with Reward Version 2
# ============================================================

env_eval_v2 = VideoAnomalyEnvV2(states, y_test)

state = env_eval_v2.reset()
done = False

dqn_predictions_v2 = []
true_labels_v2 = []

while not done:

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net_v2(state_tensor)

    action = torch.argmax(q_values).item()

    next_state, reward, done, true_label = env_eval_v2.step(action)

    dqn_predictions_v2.append(action)
    true_labels_v2.append(true_label)

    state = next_state

dqn_predictions_v2 = np.array(dqn_predictions_v2)
true_labels_v2 = np.array(true_labels_v2)

print("Predicted normal :", np.sum(dqn_predictions_v2 == 0))
print("Predicted anomaly:", np.sum(dqn_predictions_v2 == 1))


# In[25]:


# ============================================================
# Section 23: Evaluate DQN V2 Performance
# ============================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report

acc_v2 = accuracy_score(true_labels_v2, dqn_predictions_v2)
pre_v2 = precision_score(true_labels_v2, dqn_predictions_v2, zero_division=0)
rec_v2 = recall_score(true_labels_v2, dqn_predictions_v2, zero_division=0)
f1_v2 = f1_score(true_labels_v2, dqn_predictions_v2, zero_division=0)
cm_v2 = confusion_matrix(true_labels_v2, dqn_predictions_v2)

print("Accuracy :", acc_v2)
print("Precision:", pre_v2)
print("Recall   :", rec_v2)
print("F1-score :", f1_v2)

print("\nConfusion Matrix:")
print(cm_v2)

print("\nClassification Report:")
print(classification_report(true_labels_v2, dqn_predictions_v2, zero_division=0))


# In[26]:


#Reward Version 3


# In[35]:


# ============================================================
# Section 24: Define RL Environment with Reward Version 3
# ============================================================

class VideoAnomalyEnvV3:

    def __init__(self, states, labels):

        self.states = states
        self.labels = labels
        self.current_index = 0
        self.n_samples = len(states)

    def reset(self):

        self.current_index = 0
        return self.states[self.current_index]

    def step(self, action):

        true_label = self.labels[self.current_index]

        if action == 1 and true_label == 1:
            reward = 10

        elif action == 0 and true_label == 0:
            reward = 5

        elif action == 1 and true_label == 0:
            reward = -7

        elif action == 0 and true_label == 1:
            reward = -10

        self.current_index += 1

        done = self.current_index >= self.n_samples

        if done:
            next_state = np.zeros_like(self.states[0])
        else:
            next_state = self.states[self.current_index]

        return next_state, reward, done, true_label


# In[36]:


# ============================================================
# Section 25: Reinitialize DQN for Reward Version 3
# ============================================================

policy_net_v3 = DQN(state_size, action_size).to(device)
target_net_v3 = DQN(state_size, action_size).to(device)

target_net_v3.load_state_dict(policy_net_v3.state_dict())
target_net_v3.eval()

optimizer_v3 = optim.Adam(policy_net_v3.parameters(), lr=0.001)

memory_v3 = deque(maxlen=5000)

epsilon_v3 = 1.0

print(policy_net_v3)


# In[37]:


# ============================================================
# Section 26: Define Action Selection Function for DQN V3
# ============================================================

def select_action_v3(state, epsilon_v3):

    if random.random() < epsilon_v3:
        return random.randint(0, action_size - 1)

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net_v3(state_tensor)

    return torch.argmax(q_values).item()


# In[38]:


# ============================================================
# Section 27: Define Training Step for DQN V3
# ============================================================

def train_dqn_step_v3():

    if len(memory_v3) < batch_size:
        return None

    batch = random.sample(memory_v3, batch_size)

    states_batch = torch.tensor(
        np.array([item[0] for item in batch]),
        dtype=torch.float32
    ).to(device)

    actions_batch = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long
    ).unsqueeze(1).to(device)

    rewards_batch = torch.tensor(
        [item[2] for item in batch],
        dtype=torch.float32
    ).unsqueeze(1).to(device)

    next_states_batch = torch.tensor(
        np.array([item[3] for item in batch]),
        dtype=torch.float32
    ).to(device)

    dones_batch = torch.tensor(
        [item[4] for item in batch],
        dtype=torch.float32
    ).unsqueeze(1).to(device)

    q_values = policy_net_v3(states_batch).gather(
        1,
        actions_batch
    )

    with torch.no_grad():
        next_q_values = target_net_v3(
            next_states_batch
        ).max(1)[0].unsqueeze(1)

        target_q_values = rewards_batch + gamma * next_q_values * (1 - dones_batch)

    loss = criterion(q_values, target_q_values)

    optimizer_v3.zero_grad()
    loss.backward()
    optimizer_v3.step()

    return loss.item()


# In[39]:


# ============================================================
# Section 28: Train DQN Agent with Reward Version 3
# ============================================================

env_v3 = VideoAnomalyEnvV3(states, y_test)

num_episodes_v3 = 150

episode_rewards_v3 = []
episode_losses_v3 = []

for episode in range(num_episodes_v3):

    state = env_v3.reset()
    total_reward = 0
    losses = []
    done = False

    while not done:

        action = select_action_v3(state, epsilon_v3)

        next_state, reward, done, true_label = env_v3.step(action)

        memory_v3.append(
            (state, action, reward, next_state, done)
        )

        loss = train_dqn_step_v3()

        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward

    if episode % target_update_freq == 0:
        target_net_v3.load_state_dict(policy_net_v3.state_dict())

    epsilon_v3 = max(
        epsilon_min,
        epsilon_v3 * epsilon_decay
    )

    avg_loss = np.mean(losses) if losses else 0

    episode_rewards_v3.append(total_reward)
    episode_losses_v3.append(avg_loss)

    print(
        f"Episode {episode+1}/{num_episodes_v3} | "
        f"Reward: {total_reward:.2f} | "
        f"Loss: {avg_loss:.4f} | "
        f"Epsilon: {epsilon_v3:.4f}"
    )


# In[40]:


# ============================================================
# Section 29: Evaluate DQN Agent with Reward Version 3
# ============================================================

env_eval_v3 = VideoAnomalyEnvV3(states, y_test)

state = env_eval_v3.reset()
done = False

dqn_predictions_v3 = []
true_labels_v3 = []

while not done:

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net_v3(state_tensor)

    action = torch.argmax(q_values).item()

    next_state, reward, done, true_label = env_eval_v3.step(action)

    dqn_predictions_v3.append(action)
    true_labels_v3.append(true_label)

    state = next_state

dqn_predictions_v3 = np.array(dqn_predictions_v3)
true_labels_v3 = np.array(true_labels_v3)

print("Predicted normal :", np.sum(dqn_predictions_v3 == 0))
print("Predicted anomaly:", np.sum(dqn_predictions_v3 == 1))


# In[ ]:


# ============================================================
# Section 30: Evaluate DQN V3 Performance
# ============================================================

acc_v3 = accuracy_score(true_labels_v3, dqn_predictions_v3)
pre_v3 = precision_score(true_labels_v3, dqn_predictions_v3, zero_division=0)
rec_v3 = recall_score(true_labels_v3, dqn_predictions_v3, zero_division=0)
f1_v3 = f1_score(true_labels_v3, dqn_predictions_v3, zero_division=0)
cm_v3 = confusion_matrix(true_labels_v3, dqn_predictions_v3)

print("Accuracy :", acc_v3)
print("Precision:", pre_v3)
print("Recall   :", rec_v3)
print("F1-score :", f1_v3)

print("\nConfusion Matrix:")
print(cm_v3)

print("\nClassification Report:")
print(classification_report(true_labels_v3, dqn_predictions_v3, zero_division=0))


# In[ ]:


##Save Final Results


# In[ ]:


# ============================================================
# Section 31: Save Final Experimental Results
# ============================================================

results = {
    "Autoencoder_Initial": {
        "Accuracy": 0.599,
        "Precision": 0.422,
        "Recall": 0.906,
        "F1": 0.576,
        "ROC_AUC": 0.731
    },

    "Autoencoder_Optimized": {
        "Accuracy": 0.640,
        "Precision": 0.450,
        "Recall": 0.850,
        "F1": 0.590,
        "ROC_AUC": 0.731
    },

    "DQN_V1": {
        "Accuracy": 0.633,
        "Precision": 0.445,
        "Recall": 0.892,
        "F1": 0.593
    },

    "DQN_V2": {
        "Accuracy": 0.723,
        "Precision": 0.527,
        "Recall": 0.763,
        "F1": 0.623
    },

    "DQN_V3": {
        "Accuracy": 0.744,
        "Precision": 0.559,
        "Recall": 0.705,
        "F1": 0.623
    }
}

print("Results saved in memory")


# In[ ]:


#Create Results DataFrame


# In[ ]:


# ============================================================
# Section 32: Results Table
# ============================================================

import pandas as pd

results_df = pd.DataFrame(results).T

results_df


# In[ ]:


#Save Results to Excel


# In[ ]:


# ============================================================
# Section 33: Export Results to Excel
# ============================================================

results_df.to_excel(
    project_root / "Final_Results_Avenue.xlsx"
)

print("Excel file saved.")


# In[ ]:


#Plot Comparison


# In[ ]:


# ============================================================
# Section 34: Compare Models
# ============================================================

import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1"]

results_df[metrics].plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Model Comparison on Avenue Dataset")
plt.ylabel("Score")
plt.grid(True)
plt.show()


# In[ ]:


# ============================================================
# Section 35: Save DQN V3 Model
# ============================================================

torch.save(
    model.state_dict(),
    models_dir / "autoencoder_avenue.pth"
)

torch.save(
    policy_net_v2.state_dict(),
    models_dir / "dqn_v2_avenue.pth"
)

torch.save(
    policy_net_v3.state_dict(),
    models_dir / "dqn_v3_avenue.pth"
)

print("DQN V3 model saved.")


# In[18]:


# ============================================================
# Figure 4-4: Comparison of DQN Reward Versions
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

dqn_results = pd.DataFrame({
    "DQN-V1": [0.632891, 0.444542, 0.892226, 0.593420],
    "DQN-V2": [0.723077, 0.526829, 0.763251, 0.623377],
    "DQN-V3": [0.744297, 0.558824, 0.704947, 0.623438],
}, index=["Accuracy", "Precision", "Recall", "F1-Score"])

ax = dqn_results.plot(kind="bar", figsize=(9, 5))

plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Model")

save_path = results_dir / "figure_4_4_dqn_versions_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)


# In[19]:


# ============================================================
# Figure 4-5: Comparison of Baseline and Proposed Method
# ============================================================

baseline_vs_dqn = pd.DataFrame({
    "Autoencoder + Threshold": [0.640000, 0.450000, 0.850000, 0.590000],
    "DQN-V3": [0.744297, 0.558824, 0.704947, 0.623438],
}, index=["Accuracy", "Precision", "Recall", "F1-Score"])

ax = baseline_vs_dqn.plot(kind="bar", figsize=(9, 5))

plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Model")

save_path = results_dir / "figure_4_5_baseline_vs_dqn.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)


# In[ ]:




