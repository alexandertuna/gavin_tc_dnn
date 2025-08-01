import os
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import numpy as np, torch, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np, torch, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

from ml import SiameseDataset, PLST5Dataset, EmbeddingNetT5, EmbeddingNetpLS, ContrastiveLoss

ETA_MAX = 2.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self,
                 seed: int,
                 emb_dim: int,
                 bonus_features,
                 # ------------
                 X_left_train,
                 X_left_test,
                 X_right_train,
                 X_right_test,
                 y_t5_train,
                 y_t5_test,
                 w_t5_train,
                 w_t5_test,
                 # ------------
                 X_pls_train,
                 X_pls_test,
                 X_t5raw_train,
                 X_t5raw_test,
                 y_pls_train,
                 y_pls_test,
                 w_pls_train,
                 w_pls_test,
                 ):

        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)

        def remove_bonus_features(X):
            return X[:, :-bonus_features] if bonus_features > 0 else X
        X_left_train  = remove_bonus_features(X_left_train)
        X_left_test   = remove_bonus_features(X_left_test)
        X_right_train = remove_bonus_features(X_right_train)
        X_right_test  = remove_bonus_features(X_right_test)
        X_pls_train   = remove_bonus_features(X_pls_train)
        X_pls_test    = remove_bonus_features(X_pls_test)
        X_t5raw_train = remove_bonus_features(X_t5raw_train)
        X_t5raw_test  = remove_bonus_features(X_t5raw_test)

        # store data
        self.X_left_test = X_left_test
        self.X_right_test = X_right_test
        self.X_pls_test = X_pls_test
        self.X_t5raw_test = X_t5raw_test
        self.y_t5_test = y_t5_test
        self.y_pls_test = y_pls_test

        print("Creating datasets ...")
        train_t5_ds = SiameseDataset(X_left_train, X_right_train, y_t5_train, w_t5_train)
        test_t5_ds  = SiameseDataset(X_left_test,  X_right_test,  y_t5_test,  w_t5_test)

        train_pls_ds = PLST5Dataset(X_pls_train, X_t5raw_train, y_pls_train, w_pls_train)
        test_pls_ds  = PLST5Dataset(X_pls_test,  X_t5raw_test,  y_pls_test,  w_pls_test)

        batch_size = 1024
        num_workers = min(os.cpu_count() or 4, 8)

        print("Creating loaders ...")
        self.train_t5_loader = DataLoader(train_t5_ds, batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=True)
        self.test_t5_loader  = DataLoader(test_t5_ds,  batch_size, shuffle=False,
                                          num_workers=num_workers, pin_memory=True)
        self.train_pls_loader = DataLoader(train_pls_ds, batch_size, shuffle=True,
                                           num_workers=num_workers, pin_memory=True)
        self.test_pls_loader  = DataLoader(test_pls_ds,  batch_size, shuffle=False,
                                           num_workers=num_workers, pin_memory=True)

        print("Loaders ready:",
            f"T5 train {len(train_t5_ds)}, pLS-T5 train {len(train_pls_ds)}")

        # contrastive loss (reuse)
        self.criterion = ContrastiveLoss(margin=1.0)

        # instantiate and send to GPU/CPU
        print("Creating embedding networks ...")
        self.embed_t5 = EmbeddingNetT5(emb_dim=emb_dim).to(DEVICE)
        self.embed_pls = EmbeddingNetpLS(emb_dim=emb_dim).to(DEVICE)

        # joint optimizer over both nets
        print("Creating optimizer ...")
        self.optimizer = optim.Adam(
            list(self.embed_t5.parameters()) + list(self.embed_pls.parameters()),
            lr=0.0025
        )


    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


    def train(self, num_epochs: int = 200):
        print(time.strftime("Time: %Y-%m-%d %H:%M:%S", time.localtime()))
        self.losses_t5t5 = []
        self.losses_t5pls = []

        for epoch in range(1, num_epochs+1):
            self.embed_t5.train(); self.embed_pls.train()
            total_loss = 0.0
            total_t5   = 0.0
            total_pls  = 0.0

            # zip will stop at the shorter loader; you can also use itertools.cycle if needed
            for (l, r, y0, w0), (p5, t5f, y1, w1) in zip(self.train_t5_loader, self.train_pls_loader):
                # to device
                l   = l.to(DEVICE);   r    = r.to(DEVICE)
                y0_ = y0.to(DEVICE);  w0_  = w0.to(DEVICE)
                p5  = p5.to(DEVICE);  t5f_ = t5f.to(DEVICE)
                y1_ = y1.to(DEVICE);  w1_  = w1.to(DEVICE)

                # --- T5–T5 forward & loss ---
                e_l = self.embed_t5(l);  e_r = self.embed_t5(r)
                d0 = torch.sqrt(((e_l-e_r)**2).sum(1,keepdim=True) + 1e-6)
                loss0 = self.criterion(d0, y0_, w0_)

                # --- pLS-T5 forward & loss ---
                e_p5 = self.embed_pls(p5)
                e_t5 = self.embed_t5(t5f_)
                d1 = torch.sqrt(((e_p5-e_t5)**2).sum(1,keepdim=True) + 1e-6)
                loss1 = self.criterion(d1, y1_, w1_)

                loss = loss0 + loss1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_t5  += loss0.item()
                total_pls += loss1.item()

            avg_loss   = total_loss / len(self.train_pls_loader)
            avg_t5     = total_t5   / len(self.train_t5_loader)
            avg_pls    = total_pls  / len(self.train_pls_loader)
            print(f"Epoch {epoch}/{num_epochs}:  JointLoss={avg_loss:.4f}  "
                f"T5={avg_t5:.4f}  pLS={avg_pls:.4f}")
            self.losses_t5t5.append(avg_t5)
            self.losses_t5pls.append(avg_pls)

        # disable training mode
        self.embed_pls.eval(); self.embed_t5.eval()


    def print_thresholds(self):
        with PdfPages("thresholds.pdf") as pdf:
            self.print_t5t5_thresholds(pdf)
            self.print_t5pls_thresholds(pdf)


    def print_t5t5_thresholds(self, pdf):
        print("// T5-T5 thresholds:")
        percentiles   = [80, 85, 90, 93, 95, 98, 99]   # keep this % of non‑duplicates
        eta_edges     = np.arange(0.0, 2.75, 0.25)     # |η| binning
        dr2_threshold = 1.0e-3                         # ΔR² cut

        eta_L  = self.X_left_test[:, 0] * ETA_MAX
        phi_L  = np.arctan2(self.X_left_test[:, 2], self.X_left_test[:, 1])
        eta_R  = self.X_right_test[:, 0] * ETA_MAX
        phi_R  = np.arctan2(self.X_right_test[:, 2], self.X_right_test[:, 1])

        abs_eta = np.abs(eta_L)

        deta = eta_L - eta_R
        dphi = (phi_R - phi_L + np.pi) % (2*np.pi) - np.pi
        dR2  = deta**2 + dphi**2                       # ΔR² baseline

        self.embed_t5.eval()
        with torch.no_grad():
            L = torch.from_numpy(self.X_left_test.astype(np.float32)).to(DEVICE)
            R = torch.from_numpy(self.X_right_test.astype(np.float32)).to(DEVICE)
            dist = torch.sqrt(((self.embed_t5(L) - self.embed_t5(R))**2).sum(dim=1) + 1e-6) \
                    .cpu().numpy()

        y_test = self.y_t5_test                              # shorthand

        cut_vals   = {p: [] for p in percentiles}
        dup_rej    = {p: [] for p in percentiles}
        dr2_eff    = []
        dr2_rejdup = []

        for lo, hi in zip(eta_edges[:-1], eta_edges[1:]):
            nnd = (abs_eta >= lo) & (abs_eta < hi) & (y_test == 1)   # non‑dups
            dup = (abs_eta >= lo) & (abs_eta < hi) & (y_test == 0)   # dups

            # ΔR² metrics
            dr2_eff   .append(np.mean(dR2[nnd] >= dr2_threshold)*100 if np.any(nnd) else np.nan)
            dr2_rejdup.append(np.mean(dR2[dup] <  dr2_threshold)*100 if np.any(dup) else np.nan)

            # embedding‑distance cuts
            for p in percentiles:
                cut = np.percentile(dist[nnd], 100-p) if np.any(nnd) else np.nan
                cut_vals[p].append(cut)
                dup_rej[p].append(np.mean(dist[dup] < cut)*100 if (np.any(dup) and not np.isnan(cut)) else np.nan)

        fig, ax = plt.subplots(figsize=(10,6))
        h = ax.hist2d(abs_eta[y_test==1], dist[y_test==1],
                    bins=[eta_edges, 50], norm=LogNorm())
        fig.colorbar(h[3], ax=ax, label='Counts')
        ax.set_xlabel('|η| (hit 0)')
        ax.set_ylabel('Embedding distance')
        ax.set_title('T5-T5  •  Embedding distance vs |η|  (test non‑duplicates)')

        mid_eta = eta_edges[:-1] + 0.5*np.diff(eta_edges)
        for p, clr in zip(percentiles, plt.cm.rainbow(np.linspace(0,1,len(percentiles)))):
            ax.plot(mid_eta, cut_vals[p], '-o', color=clr, label=f'{p}% retention')
        ax.legend(); ax.grid(alpha=0.3); plt.show()
        pdf.savefig()
        plt.close()

        for p in percentiles:
            cuts = ", ".join(f"{v:.4f}" if not np.isnan(v) else "nan" for v in cut_vals[p])
            rejs = ", ".join(f"{v:.2f}"  if not np.isnan(v) else "nan" for v in dup_rej[p])
            print(f"{p}%-cut:     {{ {cuts} }}")
            print(f"  {p}%-dupRej: {{ {rejs} }}")
            print()
        eff = ", ".join(f"{v:.2f}" if not np.isnan(v) else "nan" for v in dr2_eff)
        rej = ", ".join(f"{v:.2f}" if not np.isnan(v) else "nan" for v in dr2_rejdup)
        print(f"dR2-eff  (%): {{ {eff} }}")
        print(f"dR2-dupRej (%): {{ {rej} }}")


    def print_t5pls_thresholds(self, pdf):
        print("// pLS-T5 thresholds:")
        percentiles   = [80, 85, 90, 93, 95, 98, 99]
        eta_edges     = np.arange(0.0, 2.75, 0.25)
        dr2_threshold = 1.0e-3

        eta_L  = self.X_pls_test[:, 0] * 4.0                     # pLS η was stored as η/4
        phi_L  = np.arctan2(self.X_pls_test[:, 3], self.X_pls_test[:, 2])
        eta_R  = self.X_t5raw_test[:, 0] * ETA_MAX
        phi_R  = np.arctan2(self.X_t5raw_test[:, 2], self.X_t5raw_test[:, 1])

        abs_eta = np.abs(eta_L)

        deta = eta_L - eta_R
        dphi = (phi_R - phi_L + np.pi) % (2*np.pi) - np.pi
        dR2  = deta**2 + dphi**2

        self.embed_pls.eval(); self.embed_t5.eval()
        with torch.no_grad():
            L = torch.from_numpy(self.X_pls_test.astype(np.float32)).to(DEVICE)
            R = torch.from_numpy(self.X_t5raw_test.astype(np.float32)).to(DEVICE)
            dist = torch.sqrt(((self.embed_pls(L) - self.embed_t5(R))**2).sum(dim=1) + 1e-6) \
                    .cpu().numpy()

        y_test = self.y_pls_test

        cut_vals   = {p: [] for p in percentiles}
        dup_rej    = {p: [] for p in percentiles}
        dr2_eff    = []
        dr2_rejdup = []

        for lo, hi in zip(eta_edges[:-1], eta_edges[1:]):
            nnd = (abs_eta >= lo) & (abs_eta < hi) & (y_test == 1)
            dup = (abs_eta >= lo) & (abs_eta < hi) & (y_test == 0)

            dr2_eff   .append(np.mean(dR2[nnd] >= dr2_threshold)*100 if np.any(nnd) else np.nan)
            dr2_rejdup.append(np.mean(dR2[dup] <  dr2_threshold)*100 if np.any(dup) else np.nan)

            for p in percentiles:
                cut = np.percentile(dist[nnd], 100-p) if np.any(nnd) else np.nan
                cut_vals[p].append(cut)
                dup_rej[p].append(np.mean(dist[dup] < cut)*100 if (np.any(dup) and not np.isnan(cut)) else np.nan)

        fig, ax = plt.subplots(figsize=(10,6))
        h = ax.hist2d(abs_eta[y_test==1], dist[y_test==1],
                    bins=[eta_edges, 50], norm=LogNorm())
        fig.colorbar(h[3], ax=ax, label='Counts')
        ax.set_xlabel('|η| (pLS)')
        ax.set_ylabel('Embedding distance')
        ax.set_title('pLS-T5  •  Embedding distance vs |η|  (test non-duplicates)')

        mid_eta = eta_edges[:-1] + 0.5*np.diff(eta_edges)
        for p, clr in zip(percentiles, plt.cm.rainbow(np.linspace(0,1,len(percentiles)))):
            ax.plot(mid_eta, cut_vals[p], '-o', color=clr, label=f'{p}% retention')
        ax.legend(); ax.grid(alpha=0.3); plt.show()
        pdf.savefig()
        plt.close()

        for p in percentiles:
            cuts = ", ".join(f"{v:.4f}" if not np.isnan(v) else "nan" for v in cut_vals[p])
            rejs = ", ".join(f"{v:.2f}"  if not np.isnan(v) else "nan" for v in dup_rej[p])
            print(f"{p}%-cut:     {{ {cuts} }}")
            print(f"  {p}%-dupRej: {{ {rejs} }}")
            print()
        eff = ", ".join(f"{v:.2f}" if not np.isnan(v) else "nan" for v in dr2_eff)
        rej = ", ".join(f"{v:.2f}" if not np.isnan(v) else "nan" for v in dr2_rejdup)
        print(f"dR2-eff  (%): {{ {eff} }}")
        print(f"dR2-dupRej (%): {{ {rej} }}")


    def print_weights_biases(self):
        print("*"*50)
        print(time.strftime("Time: %Y-%m-%d %H:%M:%S", time.localtime()))
        print("T5 embedding model weights and biases:")
        print_model_weights_biases(self.embed_t5)
        print("*"*50)

        print("*"*50)
        print("PLS embedding model weights and biases:")
        print_model_weights_biases(self.embed_pls)
        print("*"*50)


    def save(self, path: Path):
        print(f"Saving model to {path}")
        torch.save({
            'embed_t5': self.embed_t5.state_dict(),
            'embed_pls': self.embed_pls.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)


    def load(self, path):
        print(f"Loading model from {path}")
        checkpoint = torch.load(path) # map_location=DEVICE
        self.embed_t5.load_state_dict(checkpoint['embed_t5'])
        self.embed_pls.load_state_dict(checkpoint['embed_pls'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.embed_t5.to(DEVICE)
        self.embed_pls.to(DEVICE)


def print_formatted_weights_biases(weights, biases, layer_name):
    # Print biases
    print(f"HOST_DEVICE_CONSTANT float bias_{layer_name}[{len(biases)}] = {{")
    print(", ".join(f"{b:.7f}f" for b in biases) + " };")
    print()

    # Print weights
    print(f"HOST_DEVICE_CONSTANT const float wgtT_{layer_name}[{len(weights[0])}][{len(weights)}] = {{")
    for row in weights.T:
        formatted_row = ", ".join(f"{w:.7f}f" for w in row)
        print(f"{{ {formatted_row} }},")
    print("};")
    print()


def print_model_weights_biases(model):
    # Make sure the model is in evaluation mode
    model.eval()

    # Iterate through all named modules in the model
    for name, module in model.named_modules():
        # Check if the module is a linear layer
        if isinstance(module, nn.Linear):
            # Get weights and biases
            weights = module.weight.data.cpu().numpy()
            biases = module.bias.data.cpu().numpy()

            # Print formatted weights and biases
            print_formatted_weights_biases(weights, biases, name.replace('.', '_'))