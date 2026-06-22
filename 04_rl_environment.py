#!/usr/bin/env python
# coding: utf-8

# In[1]:


#State      = (16,512)

#Action 0   = Normal

#Action 1   = Anomaly

#Reward     = TP, TN, FP, FN

#Episode    = Video Windows

#Load Feature Files
#↓
#Build RL Environment
#↓
#State = 16 × 512
#↓
#Action = Normal / Anomaly
#↓
#Reward = Correct / Wrong Decision


# In[2]:


# ============================================================
# Section 1: Inspect Avenue Ground Truth Files
# ============================================================

from pathlib import Path

gt_dir = Path(
    r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\dataset\Avenue\testing_vol"
)

gt_files = sorted(gt_dir.glob("*"))

print("Number of ground truth files:", len(gt_files))

for f in gt_files[:10]:
    print(f.name)


# In[3]:


# ============================================================
# Section 2: Load One Ground Truth .mat File
# ============================================================

import scipy.io as sio

sample_gt_path = gt_dir / "vol01.mat"

mat_data = sio.loadmat(sample_gt_path)

print(mat_data.keys())


# In[4]:


# ============================================================
# Section 3: Inspect Ground Truth Data Structure
# ============================================================

vol = mat_data["vol"]

print("Type:", type(vol))
print("Shape:", vol.shape)
print("Dtype:", vol.dtype)
print("Min:", vol.min())
print("Max:", vol.max())


# In[6]:


# ============================================================
# Section 3: Inspect Max Value per Frame
# ============================================================

max_values = []

for i in range(vol.shape[2]):
    max_values.append(np.max(vol[:, :, i]))

max_values = np.array(max_values)

print("Min of frame max values:", max_values.min())
print("Max of frame max values:", max_values.max())
print("Unique first 20 max values:", np.unique(max_values)[:20])
print("Unique last 20 max values:", np.unique(max_values)[-20:])


# In[7]:


# ============================================================
# Section 4: Load One Testing Label Mask File
# ============================================================

import scipy.io as sio
import numpy as np

sample_label_path = mask_dir / "1_label.mat"

label_data = sio.loadmat(sample_label_path)

print(label_data.keys())


# In[8]:


# ============================================================
# Section 5: Inspect volLabel Structure
# ============================================================

volLabel = label_data["volLabel"]

print("Type:", type(volLabel))
print("Shape:", volLabel.shape)
print("Dtype:", volLabel.dtype)

print("\nFirst element information:")

first_mask = volLabel[0,0]

print("Type:", type(first_mask))
print("Shape:", first_mask.shape)
print("Dtype:", first_mask.dtype)


# In[9]:


# ============================================================
# Section 6: Visualize First Ground Truth Mask
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.imshow(first_mask, cmap="gray")
plt.colorbar()
plt.title("First Ground Truth Mask")
plt.show()


# In[10]:


# ============================================================
# Section 7: Inspect Mask Values
# ============================================================

import numpy as np

print("Min:", np.min(first_mask))
print("Max:", np.max(first_mask))

print("Unique values:")
print(np.unique(first_mask))


# In[19]:


# ============================================================
# Section 8: Convert volLabel Masks to Frame-Level Labels
# ============================================================

frame_labels = []

for i in range(volLabel.shape[1]):
    mask = volLabel[0, i]
    label = 1 if np.max(mask) > 0 else 0
    frame_labels.append(label)

frame_labels = np.array(frame_labels)

print("Frame labels shape:", frame_labels.shape)
print("Normal frames:", np.sum(frame_labels == 0))
print("Abnormal frames:", np.sum(frame_labels == 1))
print("Unique labels:", np.unique(frame_labels))


# In[20]:


# ============================================================
# Section 9: Find Abnormal Frame Ranges
# ============================================================

abnormal_indices = np.where(frame_labels == 1)[0]

print("First abnormal frame:", abnormal_indices[0])
print("Last abnormal frame:", abnormal_indices[-1])
print("Number of abnormal frames:", len(abnormal_indices))


# In[21]:


# ============================================================
# Section 10: Create Window Labels from Frame Labels
# ============================================================

window_size = 16
stride = 8

window_labels = []

for start_idx in range(
    0,
    len(frame_labels) - window_size + 1,
    stride
):

    window_frame_labels = frame_labels[
        start_idx:start_idx + window_size
    ]

    # اگر حداقل یک فریم غیرعادی باشد
    label = 1 if np.any(window_frame_labels == 1) else 0

    window_labels.append(label)

window_labels = np.array(window_labels)

print("Window labels shape:", window_labels.shape)

print("Normal windows:",
      np.sum(window_labels == 0))

print("Abnormal windows:",
      np.sum(window_labels == 1))


# In[22]:


# ============================================================
# Section 11: Verify Window Count
# ============================================================

feature_file = np.load(
    r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\features\test\01_features.npy"
)

print("Feature windows:", feature_file.shape[0])
print("Label windows:", len(window_labels))


# In[ ]:




