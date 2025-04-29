#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import pickle
from enum import Enum

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# for unpickling
from ml import SiameseDataset, EmbeddingNet, SiameseNet, ContrastiveLoss

# constants
DPT = 0
DETA = 1
DPHI = 2
DR = 3
VARS = [DPT, DETA, DPHI, DR]

DUP = 0
NODUP = 1
SAMPLES = [DUP, NODUP]

NOZOOM = 0
ZOOM = 1
VIEWS = [NOZOOM, ZOOM]

LABEL = {
    DPT: r"$\Delta p_{T}$",
    DETA: r"$\Delta \eta$",
    DPHI: r"$\Delta \phi$",
    DR: r"$\sqrt{\Delta \eta^2 + \Delta \phi^2}$",
}

TITLE = {
    DUP: "Sample: duplicates",
    NODUP: "Sample: not duplicates",
}

BINS = {
    DPT: np.arange(-0.5, 5, 0.1),
    DETA: np.arange(0, 8, 0.1),
    DPHI: np.arange(0, 3.25, 0.05),
    DR: np.arange(0, 8, 0.1),
}

ZOOMBINS = {
    DPT: np.arange(-0.1, 1.1, 0.05),
    DETA: np.arange(0, 0.25, 0.005),
    DPHI: np.arange(0, 0.25, 0.005),
    DR: np.arange(0, 0.25, 0.01),
}

T5 = 4
PT3 = 5
PT5 = 7
PLS = 8

TYPENAME = {
    T5: "T5",
    PT3: "PT3",
    PT5: "PT5",
    PLS: "PLS",
}

ETA_LO, ETA_HI = -2, 2
PHI_LO, PHI_HI = -3.1415926, 3.1415926

INPUT_TITLES = [
    "log10($p_{T}$ [GeV])",
    r"$\eta$ (scaled)",
    r"$\phi$ (scaled)",
    "candidate type (scaled)",
]

SCALED_INPUT_BINS = [
    np.arange(-0.1, 2.1, 0.05),
    np.arange(-1.1, 1.2, 0.05),
    np.arange(-1.1, 1.2, 0.05),
    np.arange(-1.1, 1.2, 0.2),
]

EMBEDDING_BINS = np.arange(-15, 15, 0.1)


def main():
    train_path = "train.pkl"
    pdf_path = "plot.pdf"
    plotter = Plotter(pdf_path, train_path)
    plotter.plot()


class Plotter():

    def __init__(self, pdf_path, pkl_path):
        self.pdf_path = pdf_path
        self.pkl_path = pkl_path
        with open(pkl_path, "rb") as fi:
            self.data = pickle.load(fi)
        self.flatten_stuff()


    def flatten_stuff(self):
        test_loader = self.data["test_loader"]
        model = self.data["model"]

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
                dpts.append(pt_l - pt_r)
                detas.append(eta_l - eta_r)
                dphis.append(dangle(phi_l, phi_r))
                drs.append( ((eta_l - eta_r)**2 + (dangle(phi_l, phi_r))**2) ** 0.5 )
                labels.append(batch_label.cpu().numpy())
                distances.append(ds)

        self.quants = {
            DPT: np.abs(flatten(dpts)),
            DETA: np.abs(flatten(detas)),
            DPHI: np.abs(flatten(dphis)),
            DR: np.abs(flatten(drs)),
        }
        self.distances = flatten(distances)

        self.labels = np.abs(flatten(labels))
        self.masks = {
            DUP: self.labels == 0,
            NODUP: self.labels == 1,
        }


    def plot(self):
        with PdfPages(self.pdf_path) as pdf:
            self.plot_inputs_vs_embeddings(pdf)
            # for view in VIEWS:
            #     self.plot_derived(pdf, view)
            # for view in VIEWS:
            #     self.plot_derived_vs_embedding(pdf, view)
            # for view in VIEWS:
            #     # scan pt
            #     # convert -delay 20 -loop 0 pt/*.png pt_animation.gif
            #     type = PT5
            #     # for pt in np.arange(0.5, 10, 0.1):
            #     for pt in [0.5, 1.0, 2.0, 4.0, 8.0]:
            #         img_name = f"pt/pt_{pt}.png"
            #         self.plot_measured_distances(pdf, view, pt, pt, type, type, img_name)

            #     # scan types
            #     pt = 1.0
            #     type_l = PT5
            #     for type_r in [PT5, PT3, PLS, T5]:
            #         self.plot_measured_distances(pdf, view, pt, pt, type_l, type_r)


    def plot_inputs_vs_embeddings(self, pdf):
        n_inp = 4
        n_emb = 4

        features = self.data["features"]
        embedding_net = self.data["embedding_net"]

        print("len(features)", len(features))
        event_idxs = slice(0, 175)
        event_features = torch.tensor(np.concat(features[event_idxs]).astype(np.float32))
        with torch.no_grad():
            event_embeddings = embedding_net(event_features).cpu().numpy()

        fig, ax = plt.subplots(ncols=n_inp, figsize=(16, 4))
        for idx in range(n_inp):
            _, _, _ = ax[idx].hist(event_features[:, idx], bins=SCALED_INPUT_BINS[idx])
            ax[idx].set_xlabel(INPUT_TITLES[idx])
            ax[idx].set_ylabel(f"Track candidates")
            ax[idx].set_title(f"All input data (ttbar, PU200)")
            # if idx == 1:
            #     print("1D eta distribution", n)
        fig.subplots_adjust(hspace=0.3, wspace=0.35, left=0.05, right=0.95)
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots(ncols=n_emb, figsize=(16, 4))
        for idx in range(n_emb):
            ax[idx].hist(event_embeddings[:, idx], bins=EMBEDDING_BINS)
            ax[idx].set_xlabel(f"Embedding dimension {idx}")
            ax[idx].set_ylabel(f"Track candidates")
            ax[idx].set_title(f"All input data (ttbar, PU200)")
        fig.subplots_adjust(hspace=0.3, wspace=0.35, left=0.05, right=0.95)
        pdf.savefig()
        plt.close()

        for inp in range(n_inp):
            fig, ax = plt.subplots(ncols=n_emb, figsize=(16, 4))
            for emb in range(n_emb):
                _, _, _, im = ax[emb].hist2d(event_features[:, inp],
                                             event_embeddings[:, emb],
                                             bins=[SCALED_INPUT_BINS[inp], EMBEDDING_BINS],
                                             cmin=0.5,
                                             )
                # fig.colorbar(im, ax=ax[emb], label="Track candidates")
                # if inp == 1:
                #     h = np.nan_to_num(h, copy=False)
                #     h_project_x = np.sum(h, axis=1)
                #     #print("h_project_x", h_project_x)

                ax[emb].set_xlabel(INPUT_TITLES[inp])
                ax[emb].set_ylabel(f"Embedding dimension {emb}")
                ax[emb].set_title(f"All input data (ttbar, PU200)")
            fig.subplots_adjust(wspace=0.35, left=0.05, right=0.95)
            pdf.savefig()
            plt.close()


    def plot_derived(self, pdf, view):
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))
        bins = BINS if view == NOZOOM else ZOOMBINS
        for sample in SAMPLES:
            mask = self.masks[sample]
            for var in VARS:
                ax[sample, var].hist(self.quants[var][mask], bins=bins[var])
                ax[sample, var].set_xlabel(LABEL[var])
                ax[sample, var].set_ylabel("Track candidates")
                ax[sample, var].set_title(TITLE[sample])
        fig.subplots_adjust(hspace=0.3, wspace=0.35, left=0.05, right=0.95)
        pdf.savefig()
        plt.close()


    def plot_derived_vs_embedding(self, pdf, view):

        dbins = np.arange(0, 20, 0.1) if view == NOZOOM else np.arange(0, 1, 0.005)
        bins = BINS if view == NOZOOM else ZOOMBINS

        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))
        for sample in SAMPLES:
            mask = self.masks[sample]
            for var in VARS:
                ax[sample, var].hist2d(self.quants[var][mask], self.distances[mask], cmin=0.5, bins=[bins[var], dbins])
                ax[sample, var].set_xlabel(LABEL[var])
                ax[sample, var].set_ylabel("Distance in embedding space")
                ax[sample, var].set_title(TITLE[sample])
        fig.subplots_adjust(hspace=0.3, wspace=0.35, left=0.05, right=0.95)
        pdf.savefig()
        plt.close()


    def plot_measured_distances(self,
                                pdf,
                                view,
                                pt_l,
                                pt_r,
                                type_l,
                                type_r,
                                img_name=None,
                                ):
        # generate a fake dataset to measure embedded distances
        bins = BINS if view == NOZOOM else ZOOMBINS
        vmin, vmax = (0.001, 20) if view == NOZOOM else (0.0, 0.6)
        ntracks = 100
        values = np.zeros(shape=(len(bins[DETA]), len(bins[DPHI])))
        batch_l = np.zeros(shape=(ntracks, 4))
        batch_r = np.zeros(shape=(ntracks, 4))
        batch_l[:, 0] = pt_l
        batch_r[:, 0] = pt_r
        batch_l[:, 3] = type_l
        batch_r[:, 3] = type_r

        for i_eta, deta in enumerate(bins[DETA]):
            eta_mid = np.random.uniform(ETA_LO, ETA_HI, ntracks)
            for i_phi, dphi in enumerate(bins[DPHI]):
                phi_mid = np.random.uniform(PHI_LO, PHI_HI, ntracks)
                # make a random sample of -1 or 1 
                eta_pm = np.random.choice([-1, 1], ntracks)
                phi_pm = np.random.choice([-1, 1], ntracks)
                batch_l[:, 1] = eta_mid + eta_pm * deta / 2
                batch_r[:, 1] = eta_mid - eta_pm * deta / 2
                batch_l[:, 2] = phi_mid + phi_pm * dphi / 2
                batch_r[:, 2] = phi_mid - phi_pm * dphi / 2
                norm_l = normalize_batch(batch_l)
                norm_r = normalize_batch(batch_r)
                with torch.no_grad():
                    distances = self.data["model"](torch.tensor(norm_l, dtype=torch.float32),
                                                   torch.tensor(norm_r, dtype=torch.float32))
                values[i_eta, i_phi] = distances.mean().item()

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(values,
                       origin='lower', aspect='auto',
                       extent=[bins[DPHI][0],
                               bins[DPHI][-1],
                               bins[DETA][0],
                               bins[DETA][-1]],
                        vmin=vmin,
                        vmax=vmax,
                        cmap="gist_rainbow")
        ax.set_xlabel(LABEL[DPHI])
        ax.set_ylabel(LABEL[DETA])
        title_pt = f"$p_T(L) = {pt_l}$ GeV, $p_T(R) = {pt_r}$ GeV"
        title_type = f"$T_L$ = {TYPENAME[type_l]}, $T_R$ = {TYPENAME[type_r]}"
        ax.set_title(f"{title_pt}, {title_type}")
        fig.colorbar(im, ax=ax, label="Distance in embedding space")
        pdf.savefig()
        if img_name:
            plt.savefig(img_name)
        plt.close()




def normalize_pt(x):
    return np.log10(x)

def normalize_eta(x):
    return x / 4.0

def normalize_phi(x):
    return x / 3.1415926

def normalize_type(x):
    return (x - 6.0) / 2.0

def normalize_batch(batch):
    normed = np.zeros_like(batch)
    normed[:, 0] = normalize_pt(batch[:, 0])
    normed[:, 1] = normalize_eta(batch[:, 1])
    normed[:, 2] = normalize_phi(batch[:, 2])
    normed[:, 3] = normalize_type(batch[:, 3])
    return normed

def dangle(x, y):
    return np.min([(2 * np.pi) - np.abs(x - y), np.abs(x - y)], axis=0)

def flatten(li):
    return np.concatenate(li).flatten()


if __name__ == "__main__":
    main()
