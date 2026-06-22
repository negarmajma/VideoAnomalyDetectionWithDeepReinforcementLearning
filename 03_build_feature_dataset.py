#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Train Videos (16)
#      ↓
#Sliding Window
#      ↓
#ResNet18
#      ↓
#Feature Files
#      ↓
#features/train/*.npy


# In[2]:


# ============================================================
# Section 1: Import Libraries
# ============================================================

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn


# In[3]:


# ============================================================
# Section 2: GPU Configuration
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)


# In[4]:


# ============================================================
# Section 3: Load ResNet18 Feature Extractor
# ============================================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


# In[5]:


# ============================================================
# Section 4: Define Image Transformations
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# In[6]:


# ============================================================
# Section 5: Define Sliding Window Function
# ============================================================

def create_sliding_windows(frame_files, window_size=16, stride=8):
    windows = []

    for start_idx in range(0, len(frame_files) - window_size + 1, stride):
        window = frame_files[start_idx:start_idx + window_size]
        windows.append(window)

    return windows


# In[7]:


# ============================================================
# Section 6: Define Feature Extraction Function
# ============================================================

def extract_window_features(window_frame_paths, transform, feature_extractor, device):
    frame_tensors = []

    for frame_path in window_frame_paths:
        img = Image.open(frame_path).convert("RGB")
        img_tensor = transform(img)
        frame_tensors.append(img_tensor)

    batch_tensor = torch.stack(frame_tensors).to(device)

    with torch.no_grad():
        features = feature_extractor(batch_tensor)

    features = features.view(features.size(0), -1)

    return features.cpu().numpy()


# In[8]:


# ============================================================
# Section 7: Extract and Save Features for All Training Videos
# ============================================================

frames_train_dir = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\frames\train")
features_train_dir = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\features\train")

features_train_dir.mkdir(parents=True, exist_ok=True)

train_video_dirs = sorted([p for p in frames_train_dir.iterdir() if p.is_dir()])

print("Number of training videos:", len(train_video_dirs))

for video_dir in train_video_dirs:
    video_name = video_dir.name
    save_path = features_train_dir / f"{video_name}_features.npy"

    if save_path.exists():
        print(f"Skipped {video_name}: already exists")
        continue

    frame_files = sorted(video_dir.glob("*.jpg"))
    windows = create_sliding_windows(frame_files, window_size=16, stride=8)

    video_features = []

    print(f"\nProcessing train video {video_name}")
    print("Frames:", len(frame_files))
    print("Windows:", len(windows))

    for i, window in enumerate(windows):
        features = extract_window_features(
            window,
            transform,
            feature_extractor,
            device
        )
        video_features.append(features)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(windows)} windows")

    video_features = np.array(video_features)
    np.save(save_path, video_features)

    print("Saved:", save_path)
    print("Feature shape:", video_features.shape)


# In[9]:


# ============================================================
# Section 8: Extract and Save Features for All Testing Videos
# ============================================================

frames_test_dir = Path(
    r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\frames\test"
)

features_test_dir = Path(
    r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\features\test"
)

features_test_dir.mkdir(parents=True, exist_ok=True)

test_video_dirs = sorted(
    [p for p in frames_test_dir.iterdir() if p.is_dir()]
)

print("Number of testing videos:", len(test_video_dirs))

for video_dir in test_video_dirs:

    video_name = video_dir.name

    save_path = features_test_dir / f"{video_name}_features.npy"

    if save_path.exists():
        print(f"Skipped {video_name}: already exists")
        continue

    frame_files = sorted(video_dir.glob("*.jpg"))

    windows = create_sliding_windows(
        frame_files,
        window_size=16,
        stride=8
    )

    video_features = []

    print(f"\nProcessing test video {video_name}")
    print("Frames:", len(frame_files))
    print("Windows:", len(windows))

    for i, window in enumerate(windows):

        features = extract_window_features(
            window,
            transform,
            feature_extractor,
            device
        )

        video_features.append(features)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(windows)} windows")

    video_features = np.array(video_features)

    np.save(save_path, video_features)

    print("Saved:", save_path)
    print("Feature shape:", video_features.shape)


# In[ ]:




