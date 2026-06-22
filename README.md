Video Anomaly Detection using Deep Reinforcement Learning

Overview
This repository contains the implementation of a video anomaly detection framework based on Deep Learning and Deep Reinforcement Learning (DRL). The proposed approach combines feature extraction, anomaly scoring, and reinforcement learning-based decision making to identify abnormal events in surveillance videos.

Research Objective
The primary objective of this project is to investigate whether Deep Reinforcement Learning can improve anomaly detection performance compared with traditional threshold-based decision methods.

Proposed Framework
1. Video preprocessing
2. Feature extraction using ResNet18
3. Normal behavior modeling using Autoencoder
4. Reinforcement learning decision making using DQN
5. Performance evaluation

Dataset
Avenue Dataset (CUHK)

Experimental Results
Autoencoder Baseline:
ROC-AUC: 0.7309
Accuracy: 64.0%
F1-Score: 59.0%

Proposed DQN Framework (DQN-V3):
Accuracy: 74.43%
Precision: 55.88%
Recall: 70.49%
F1-Score: 62.34%

Technologies
Python, PyTorch, NumPy, OpenCV, Scikit-Learn, Matplotlib, Jupyter Notebook

Hardware
GPU: NVIDIA RTX 4060 Laptop GPU
CPU: Intel Core i7-12700H
RAM: 16 GB

Future Work
PPO, A3C, Vision Transformers, Multi-dataset evaluation, Online anomaly detection.

Negar Majmae

