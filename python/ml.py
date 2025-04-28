#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SiameseDataset(Dataset):
    def __init__(self, X_left, X_right, y):
        self.X_left = X_left.astype(np.float32)
        self.X_right = X_right.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_left[idx], self.X_right[idx], self.y[idx]


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim=4, embedding_dim=2):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)  # New hidden layer with 16 units
        self.fc3 = nn.Linear(16, embedding_dim)  # Final output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        # Compute Euclidean distance between the two embeddings.
        distance = torch.sqrt(torch.sum((out1 - out2)**2, dim=1, keepdim=True) + 1e-6)
        return distance


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # For duplicate pairs (label 0): loss = distance^2.
        # For non-duplicates (label 1): loss = max(margin - distance, 0)^2.
        loss_similar = (1 - label) * torch.pow(distance, 2)
        loss_dissimilar = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss

