#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import pickle

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


def main():

    # Load the data
    with open("model_data.pkl", "rb") as fi:
        di = pickle.load(fi)

    pdf_path = "plots.pdf"
    pdf = PdfPages(pdf_path)

    # alex plots
    # print("features:", type(features))
    # print("embedding_net:", type(embedding_net))
    # print("test_loader:", type(test_loader))
    # print("model:", type(model))
    plot_inputs_vs_embeddings(di["features"], di["embedding_net"], pdf)
    plot_derived_vs_embedding(di["test_loader"], di["model"], pdf)

    # Save the PDF file
    pdf.close()


def plot_inputs_vs_embeddings(features, embedding_net, pdf):
    n_inp = 4
    n_emb = 4
    
    # event_idx = 0
    # event_features = torch.tensor(features[event_idx].astype(np.float32))
    print("len(features)", len(features))
    event_idxs = slice(0, 175)
    event_features = torch.tensor(np.concat(features[event_idxs]).astype(np.float32))
    with torch.no_grad():
        event_embeddings = embedding_net(event_features).cpu().numpy()
    print(event_features.shape)
    print(event_embeddings.shape)

    titles_inp = [
        "log10(pT)",
        "eta (scaled)",
        "phi (scaled)",
        "type (scaled)",
    ]
    bins_inp = [
        np.arange(-0.1, 2.1, 0.05),
        np.arange(-1.1, 1.2, 0.05),
        np.arange(-1.1, 1.2, 0.05),
        np.arange(-1.1, 1.2, 0.2),
    ]
    bins_emb = [
        np.arange(-15, 15, 0.1),
        np.arange(-15, 15, 0.1),
        np.arange(-15, 15, 0.1),
        np.arange(-15, 15, 0.1),
        # np.arange(-5, 15, 0.1),
        # np.arange(-2, 5, 0.1),
        # np.arange(-11, 6, 0.1),
        # np.arange(-1, 3, 0.05),
    ]
    
    fig, ax = plt.subplots(ncols=n_inp, figsize=(16, 4))
    for idx in range(n_inp):
        ax[idx].hist(event_features[:, idx], bins=bins_inp[idx])
        ax[idx].set_xlabel(titles_inp[idx])
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(ncols=n_emb, figsize=(16, 4))
    for idx in range(n_emb):
        ax[idx].hist(event_embeddings[:, idx], bins=bins_emb[idx])
        ax[idx].set_xlabel(f"Embedding dimension {idx}")
    pdf.savefig()
    plt.close()

    for inp in range(n_inp):
        fig, ax = plt.subplots(ncols=n_emb, figsize=(16, 4))
        for emb in range(n_emb):
            _, _, _, im = ax[emb].hist2d(event_features[:, inp],
                                         event_embeddings[:, emb],
                                         bins=[bins_inp[inp], bins_emb[emb]],
                                         cmin=0.5,
                                         )
            ax[emb].set_xlabel(titles_inp[inp])
            ax[emb].set_ylabel(f"Embedding dimension {emb}")
            ax[emb].set_title(f"All input data (ttbar, PU200)")
        fig.subplots_adjust(wspace=0.35, left=0.05, right=0.95)
        pdf.savefig()
        plt.close()


def dangle(x, y):
    return np.min([(2 * np.pi) - np.abs(x - y), np.abs(x - y)], axis=0)

def plot_derived_vs_embedding(test_loader, model, pdf):

    distances, labels, dpts, detas, dphis, drs = [], [], [], [], [], []
    with torch.no_grad():
        for batch_left, batch_right, batch_label in test_loader:
            ds = model(batch_left, batch_right)
            pt_l = (10 ** batch_left[:, 0]).cpu().numpy()
            pt_r = (10 ** batch_right[:, 0]).cpu().numpy()
            eta_l = (4 * batch_left[:, 1]).cpu().numpy()
            eta_r = (4 * batch_right[:, 1]).cpu().numpy()
            phi_l = (3.1415926 * batch_left[:, 2]).cpu().numpy()
            phi_r = (3.1415926 * batch_right[:, 2]).cpu().numpy()
            # print(batch_left.shape)
            dpts.append(pt_l - pt_r)
            detas.append(eta_l - eta_r)
            dphis.append(dangle(phi_l, phi_r))
            drs.append( ((eta_l - eta_r)**2 + (dangle(phi_l, phi_r))**2) ** 0.5 )
            labels.append(batch_label.cpu().numpy())
            distances.append(ds)
            # break

    def flatten(li):
        return np.concatenate(li).flatten()

    distances = flatten(distances)
    labels = np.abs(flatten(labels))
    dpts = np.abs(flatten(dpts))
    detas = np.abs(flatten(detas))
    dphis = np.abs(flatten(dphis))
    drs = np.abs(flatten(drs))

    masks = [
        labels == 0,
        labels == 1,
    ]
    bins_distance = np.arange(0, 20, 0.1)
    bins_dpt = np.arange(-1, 5, 0.1)
    bins_deta = np.arange(0, 7, 0.1)
    bins_dphi = np.arange(0, 3.25, 0.05)
    bins_dr = np.arange(0, 7, 0.1)
    
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))
    for idx, mask in enumerate(masks):
        ax[idx, 0].hist(dpts[mask], bins=bins_dpt)
        ax[idx, 1].hist(detas[mask], bins=bins_deta)
        ax[idx, 2].hist(dphis[mask], bins=bins_dphi)
        ax[idx, 3].hist(drs[mask], bins=bins_dr)
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))
    for idx, mask in enumerate(masks):
        ax[idx, 0].hist2d(dpts[mask], distances[mask], cmin=0.5, bins=[bins_dpt, bins_distance])
        ax[idx, 1].hist2d(detas[mask], distances[mask], cmin=0.5, bins=[bins_deta, bins_distance])
        ax[idx, 2].hist2d(dphis[mask], distances[mask], cmin=0.5, bins=[bins_dphi, bins_distance])
        ax[idx, 3].hist2d(drs[mask], distances[mask], cmin=0.5, bins=[bins_dr, bins_distance])
    pdf.savefig()
    plt.close()



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


if __name__ == "__main__":
    main()
