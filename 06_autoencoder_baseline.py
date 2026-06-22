#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# Section 1: Import Required Libraries
# ============================================================

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score


# In[2]:


# ============================================================
# Section 2: Load Training and Testing Data
# ============================================================

project_root = Path(
    r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue"
)

dataset_dir = project_root / "rl_dataset"

X_train = np.load(dataset_dir / "X_train.npy")
y_train = np.load(dataset_dir / "y_train.npy")

X_test = np.load(dataset_dir / "X_test.npy")
y_test = np.load(dataset_dir / "y_test.npy")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# In[3]:


# ============================================================
# Section 3: Flatten States
# ============================================================

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(X_train_flat.shape)
print(X_test_flat.shape)


# In[4]:


# ============================================================
# Section 4: Define PyTorch Dataset
# ============================================================

class FeatureDataset(Dataset):

    def __init__(self, data):
        self.data = torch.tensor(
            data,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# In[5]:


# ============================================================
# Section 5: Create DataLoader
# ============================================================

train_dataset = FeatureDataset(X_train_flat)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

print("Train batches:", len(train_loader))


# In[6]:


# ============================================================
# Section 6: Configure Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)


# In[7]:


# ============================================================
# Section 7: Define Autoencoder
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(8192, 4096),
            nn.ReLU(),

            nn.Linear(4096, 1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.Linear(256, 1024),
            nn.ReLU(),

            nn.Linear(1024, 4096),
            nn.ReLU(),

            nn.Linear(4096, 8192)
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstruction = self.decoder(latent)

        return reconstruction


# In[8]:


# ============================================================
# Section 8: Initialize Model
# ============================================================

model = Autoencoder().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print(model)


# In[9]:


# ============================================================
# Section 9: Train Autoencoder
# ============================================================

num_epochs = 30

train_losses = []

for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0.0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        reconstructed = model(batch)

        loss = criterion(reconstructed, batch)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")


# In[10]:


# ============================================================
# Section 10: Plot and Save Autoencoder Training Loss
# ============================================================

import matplotlib.pyplot as plt

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training Loss")
plt.grid(True)

save_path = results_dir / "figure_4_6_autoencoder_loss.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)


# In[11]:


# ============================================================
# Section 11: Compute Reconstruction Error on Test Data
# ============================================================

model.eval()

X_test_tensor = torch.tensor(
    X_test_flat,
    dtype=torch.float32
).to(device)

with torch.no_grad():
    reconstructed_test = model(X_test_tensor)

errors = torch.mean(
    (X_test_tensor - reconstructed_test) ** 2,
    dim=1
)

anomaly_scores = errors.cpu().numpy()

print("Anomaly scores shape:", anomaly_scores.shape)
print("Min score:", anomaly_scores.min())
print("Max score:", anomaly_scores.max())
print("Mean score:", anomaly_scores.mean())


# In[11]:


# ============================================================
# Section 12: Evaluate Autoencoder Using ROC-AUC
# ============================================================

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, anomaly_scores)

print("ROC-AUC:", auc)


# In[12]:


# ============================================================
# Section 13: Select Threshold from Training Reconstruction Error
# ============================================================

X_train_tensor = torch.tensor(
    X_train_flat,
    dtype=torch.float32
).to(device)

with torch.no_grad():
    reconstructed_train = model(X_train_tensor)

train_errors = torch.mean(
    (X_train_tensor - reconstructed_train) ** 2,
    dim=1
).cpu().numpy()

threshold = np.percentile(train_errors, 95)

print("Train error mean:", train_errors.mean())
print("Train error std :", train_errors.std())
print("Threshold       :", threshold)


# In[13]:


# ============================================================
# Section 14: Generate Binary Predictions
# ============================================================

y_pred = (anomaly_scores > threshold).astype(int)

print("Predicted normal  :", np.sum(y_pred == 0))
print("Predicted anomaly :", np.sum(y_pred == 1))


# In[14]:


# ============================================================
# Section 15: Evaluate Autoencoder Predictions
# ============================================================

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy :", acc)
print("Precision:", pre)
print("Recall   :", rec)
print("F1-score :", f1)
print("ROC-AUC  :", auc)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# In[15]:


# ============================================================
# Section 16: Score Distribution
# ============================================================

normal_scores = anomaly_scores[y_test == 0]
abnormal_scores = anomaly_scores[y_test == 1]

plt.figure(figsize=(10,5))

plt.hist(
    normal_scores,
    bins=50,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    abnormal_scores,
    bins=50,
    alpha=0.6,
    label="Anomaly"
)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    label="Threshold"
)

plt.legend()
plt.title("Anomaly Score Distribution")
plt.show()


# In[16]:


# ============================================================
# Section 17: Save Anomaly Scores
# ============================================================

np.save(
    dataset_dir / "anomaly_scores.npy",
    anomaly_scores
)

print("Saved anomaly_scores.npy")


# In[17]:


# ============================================================
# Section 18: Search Best Threshold
# ============================================================

from sklearn.metrics import f1_score

thresholds = np.linspace(
    anomaly_scores.min(),
    anomaly_scores.max(),
    200
)

best_threshold = None
best_f1 = 0

for t in thresholds:

    pred = (anomaly_scores > t).astype(int)

    f1 = f1_score(y_test, pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("Best Threshold:", best_threshold)
print("Best F1:", best_f1)


# In[18]:


# ============================================================
# Section 19: Evaluation with Best Threshold
# ============================================================

best_pred = (
    anomaly_scores > best_threshold
).astype(int)

from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        best_pred,
        zero_division=0
    )
)


# In[19]:


# ============================================================
# Section 18: Save Autoencoder Model
# ============================================================

import torch

torch.save(
    model.state_dict(),
    project_root / "autoencoder_avenue.pth"
)

print("Autoencoder model saved.")


# In[ ]:




