#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Frame
##  ↓
##ResNet18
##  ↓
## 512 Feature Vector


# In[2]:


# تست اتصال به GPU
import torch

print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


# In[3]:


#بارگذاری ResNet18

import torch
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model = model.to(device)
model.eval()

print("Model loaded on:", device)


# In[4]:


# استخراج ویژگی 512 بعدی از یک فریم
# خروجی طبقه بندی ImageNet 


# ============================================================
# Section 1: Import Required Libraries
# ============================================================
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch

# ============================================================
# Section 2: Load Sample Frame
# ============================================================
frame_path = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\frames\train\01\frame_00000.jpg")

# ============================================================
# Section 3: Define Image Transformations
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# Section 4: Prepare Input Tensor
# ============================================================
img = Image.open(frame_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    feature = model(input_tensor)

print(feature.shape)


# In[5]:


# حذف لایه آخر و دریافت ویژگی 512 بعدی

# ============================================================
# Section 7: Remove ResNet18 Classification Layer
# ============================================================

import torch.nn as nn

feature_extractor = nn.Sequential(*list(model.children())[:-1])

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

print(feature_extractor)


# In[6]:


# ============================================================
# Section 8: Extract 512-Dimensional Feature Vector
# ============================================================

with torch.no_grad():
    feature_512 = feature_extractor(input_tensor)

feature_512 = feature_512.view(feature_512.size(0), -1)

print("Feature shape:", feature_512.shape)


# In[7]:


# ============================================================
# Section 9: Inspect Feature Vector
# ============================================================

print(feature_512[0][:20])


# In[8]:


# ذخیره ویژگی ها 

# ============================================================
# Section 10: Save Feature Vector
# ============================================================

import numpy as np

feature_np = feature_512.cpu().numpy()

np.save("sample_feature.npy", feature_np)

print("Feature saved successfully.")


# In[9]:


# ============================================================
# Section 11: Create a 16-Frame Sliding Window from One Video
# ============================================================

from pathlib import Path

video_frame_dir = Path(
    r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\frames\train\01"
)

frame_files = sorted(video_frame_dir.glob("*.jpg"))

window_size = 16
stride = 8   # یعنی هر پنجره 8 فریم جلو می‌رود

windows = []

for start_idx in range(0, len(frame_files) - window_size + 1, stride):
    window = frame_files[start_idx:start_idx + window_size]
    windows.append(window)

print("Total frames:", len(frame_files))
print("Total 16-frame windows:", len(windows))
print("First window:")
for f in windows[0]:
    print(f.name)


# In[10]:


# ============================================================
# Section 12: Extract Features from One 16-Frame Window
# ============================================================

import torch
import numpy as np
from PIL import Image

sample_window = windows[0]

frame_tensors = []

for frame_path in sample_window:
    img = Image.open(frame_path).convert("RGB")
    img_tensor = transform(img)
    frame_tensors.append(img_tensor)

batch_tensor = torch.stack(frame_tensors).to(device)

print("Input batch shape:", batch_tensor.shape)

with torch.no_grad():
    window_features = feature_extractor(batch_tensor)

window_features = window_features.view(window_features.size(0), -1)

print("Window feature shape:", window_features.shape)


# In[11]:


# ============================================================
# Section 13: Define Function to Extract Window Features
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


# In[12]:


# ============================================================
# Section 14: Test Feature Extraction Function on First 3 Windows
# ============================================================

test_features = []

for i in range(3):
    features = extract_window_features(
        windows[i],
        transform,
        feature_extractor,
        device
    )
    test_features.append(features)

test_features = np.array(test_features)

print("Test features shape:", test_features.shape)


# In[13]:


# ذخیره ویژگی های کل ویدئو Train/01
# ============================================================
# Section 15: Extract and Save Features for One Video
# ============================================================

output_dir = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\features\train")
output_dir.mkdir(parents=True, exist_ok=True)

video_name = "01"
video_features = []

for i, window in enumerate(windows):
    features = extract_window_features(
        window,
        transform,
        feature_extractor,
        device
    )
    video_features.append(features)

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(windows)} windows")

video_features = np.array(video_features)

save_path = output_dir / f"{video_name}_features.npy"
np.save(save_path, video_features)

print("Saved:", save_path)
print("Video feature shape:", video_features.shape)


# In[ ]:




