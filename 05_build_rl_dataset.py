#!/usr/bin/env python
# coding: utf-8

# In[1]:


#21 فایل label
#+
#21 فایل feature
#↓
#RL Dataset

#features/test/*.npy
#+
#Ground_truth_demo/testing_label_mask/*.mat
#↓
#X_test.npy
#y_test.npy


# In[2]:


# ============================================================
# Section 1: Import Required Libraries
# ============================================================

from pathlib import Path
import numpy as np
import scipy.io as sio


# In[3]:


# ============================================================
# Section 2: Define Project Paths
# ============================================================

project_root = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue")

features_test_dir = project_root / "features" / "test"
mask_dir = project_root / "Ground_truth_demo" / "testing_label_mask"

output_dir = project_root / "rl_dataset"
output_dir.mkdir(parents=True, exist_ok=True)

print("Features path:", features_test_dir)
print("Mask path:", mask_dir)
print("Output path:", output_dir)


# In[4]:


# ============================================================
# Section 3: Define Function to Convert Frame Masks to Frame Labels
# ============================================================

def load_frame_labels_from_mat(mat_path):
    label_data = sio.loadmat(mat_path)
    volLabel = label_data["volLabel"]

    frame_labels = []

    for i in range(volLabel.shape[1]):
        mask = volLabel[0, i]
        label = 1 if np.max(mask) > 0 else 0
        frame_labels.append(label)

    return np.array(frame_labels)


# In[5]:


# ============================================================
# Section 4: Define Function to Convert Frame Labels to Window Labels
# ============================================================

def create_window_labels(frame_labels, window_size=16, stride=8):
    window_labels = []

    for start_idx in range(0, len(frame_labels) - window_size + 1, stride):
        window = frame_labels[start_idx:start_idx + window_size]
        label = 1 if np.any(window == 1) else 0
        window_labels.append(label)

    return np.array(window_labels)


# In[6]:


# ============================================================
# Section 5: Build RL Test Dataset from Features and Ground Truth
# ============================================================

X_test_list = []
y_test_list = []

for video_id in range(1, 22):
    feature_file = features_test_dir / f"{video_id:02d}_features.npy"
    label_file = mask_dir / f"{video_id}_label.mat"

    features = np.load(feature_file)
    frame_labels = load_frame_labels_from_mat(label_file)
    window_labels = create_window_labels(
        frame_labels,
        window_size=16,
        stride=8
    )

    print(f"Video {video_id:02d}")
    print("Features:", features.shape)
    print("Labels:", window_labels.shape)

    if features.shape[0] != window_labels.shape[0]:
        print("Mismatch found!")
        break

    X_test_list.append(features)
    y_test_list.append(window_labels)

X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print("Final X_test shape:", X_test.shape)
print("Final y_test shape:", y_test.shape)
print("Normal windows:", np.sum(y_test == 0))
print("Abnormal windows:", np.sum(y_test == 1))


# In[7]:


# ============================================================
# Section 6: Save RL Test Dataset
# ============================================================

np.save(output_dir / "X_test.npy", X_test)
np.save(output_dir / "y_test.npy", y_test)

print("Saved X_test.npy and y_test.npy")


# In[8]:


# ============================================================
# Section 7: Build RL Training Dataset from Normal Training Features
# ============================================================

features_train_dir = project_root / "features" / "train"

X_train_list = []
y_train_list = []

train_feature_files = sorted(features_train_dir.glob("*_features.npy"))

print("Number of train feature files:", len(train_feature_files))

for feature_file in train_feature_files:
    features = np.load(feature_file)

    # Avenue training videos are normal
    labels = np.zeros(features.shape[0], dtype=np.int64)

    print(feature_file.name)
    print("Features:", features.shape)
    print("Labels:", labels.shape)

    X_train_list.append(features)
    y_train_list.append(labels)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

print("Final X_train shape:", X_train.shape)
print("Final y_train shape:", y_train.shape)
print("Normal train windows:", np.sum(y_train == 0))
print("Abnormal train windows:", np.sum(y_train == 1))


# In[9]:


# ============================================================
# Section 8: Save RL Training Dataset
# ============================================================

np.save(output_dir / "X_train.npy", X_train)
np.save(output_dir / "y_train.npy", y_train)

print("Saved X_train.npy and y_train.npy")


# In[ ]:




