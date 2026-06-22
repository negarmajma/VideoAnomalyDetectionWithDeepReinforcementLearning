#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# Section: Create Figure 3-2 from Avenue Dataset
# Normal and Abnormal Frame Samples
# ============================================================

from pathlib import Path
import cv2
import matplotlib.pyplot as plt

project_root = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue")

# Normal samples from training videos
normal_frames = [
    project_root / "frames" / "train" / "01" / "frame_00050.jpg",
    project_root / "frames" / "train" / "02" / "frame_00200.jpg",
    project_root / "frames" / "train" / "03" / "frame_00400.jpg",
]

# Abnormal samples from test video 01
# Based on previous analysis: abnormal frames start around frame 77
abnormal_frames = [
    project_root / "frames" / "test" / "01" / "frame_00090.jpg",
    project_root / "frames" / "test" / "01" / "frame_00120.jpg",
    project_root / "frames" / "test" / "01" / "frame_00200.jpg",
]

all_frames = normal_frames + abnormal_frames
titles = [
    "Normal Frame 1",
    "Normal Frame 2",
    "Normal Frame 3",
    "Abnormal Frame 1",
    "Abnormal Frame 2",
    "Abnormal Frame 3",
]

plt.figure(figsize=(12, 7))

for i, frame_path in enumerate(all_frames):
    img = cv2.imread(str(frame_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(titles[i], fontsize=12)
    plt.axis("off")

plt.tight_layout()

save_path = project_root / "results" / "figure_3_2_avenue_samples.png"
save_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved at:", save_path)


# In[ ]:




