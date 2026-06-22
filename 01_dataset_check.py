#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[2]:


## اتصال به دیتا ست

from pathlib import Path

train_path = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\dataset\Avenue\training_videos")
test_path = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\dataset\Avenue\testing_videos")

train_files = list(train_path.glob("*"))
test_files = list(test_path.glob("*"))

print("Train videos:", len(train_files))
print("Test videos:", len(test_files))

print("\nFirst training videos:")
for f in train_files[:5]:
    print(f.name)

print("\nFirst testing videos:")
for f in test_files[:5]:
    print(f.name)


# In[3]:


# خواندن مشخصات اولین ویدئو

import cv2

video_path = str(train_files[0])

cap = cv2.VideoCapture(video_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Frames:", frame_count)
print("Resolution:", width, "x", height)
print("FPS:", fps)

cap.release()


# In[4]:


# خواندن اولین فریم

import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8,6))
plt.imshow(frame)
plt.axis("off")
plt.show()

cap.release()


# In[5]:


# استخراج فریم از ویدئو

from pathlib import Path
import cv2

base_path = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\dataset\Avenue")

video_folders = {
    "train": base_path / "training_videos",
    "test": base_path / "testing_videos"
}

output_base = Path(r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\frames")

for split, video_dir in video_folders.items():
    output_split = output_base / split
    output_split.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.glob("*.avi"))

    for video_path in video_files:
        video_name = video_path.stem
        output_video_dir = output_split / video_name
        output_video_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_file = output_video_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_file), frame)

            frame_idx += 1

        cap.release()
        print(f"{split} video {video_name}: {frame_idx} frames saved")


# In[6]:


# تست فریم ها

from pathlib import Path
import cv2
import matplotlib.pyplot as plt

frame_path = Path(
    r"D:\imp-ANOMALYDETECTION\Video_anomaly_avenue\frames\train\01\frame_00000.jpg"
)

img = cv2.imread(str(frame_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)

plt.figure(figsize=(10,6))
plt.imshow(img)
plt.axis("off")
plt.show()


# In[7]:


# اعمال Resize
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(str(frame_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_resized = cv2.resize(img, (224,224))

print(img_resized.shape)

plt.figure(figsize=(6,6))
plt.imshow(img_resized)
plt.axis("off")
plt.show()


# In[8]:


img.shape


# In[9]:


img_resized.shape


# In[ ]:




