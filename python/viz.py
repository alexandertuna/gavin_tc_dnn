import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ETA_MAX = 2.5

class Plotter:

    def __init__(self, trainer):
        self.trainer = trainer


    def plot(self, pdf_path: Path):
        with PdfPages(pdf_path) as pdf:
            self.plot_t5t5_performance(pdf)
            self.plot_loss_per_epoch(pdf)


    def plot_loss_per_epoch(self, pdf: PdfPages):
        if not hasattr(self.trainer, "losses_t5t5") or not hasattr(self.trainer, "losses_t5pls"):
            print("No losses to plot. Make sure to call train() first.")
            return
        # Plot training losses
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(self.trainer.losses_t5t5, label="T5-T5 Loss")
        ax.plot(self.trainer.losses_t5pls, label="T5-PLS Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Losses")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(top=True, right=True)
        pdf.savefig(fig)
        plt.close(fig)


    def plot_t5t5_performance(self, pdf: PdfPages):

        self.trainer.embed_t5.eval()

        # 1) collect distances and labels
        dist_t5_all = []
        dr_t5_all   = []
        lbl_t5_all  = []

        with torch.no_grad():
            for x_left, x_right, y, _ in self.trainer.test_t5_loader:
                # x_left  = x_left.to(device)
                # x_right = x_right.to(device)

                # check for identical features
                identical = (x_left == x_right).all(dim=1)
                identical_dup    = identical & (y == 0).squeeze()  # only count identical pairs that are duplicates
                identical_nondup = identical & (y == 1).squeeze()  # only count identical pairs that are duplicates
                # print(identical.shape, identical.sum().item(), "identical pairs in batch")
                # print(identical_dup.shape, identical_dup.sum().item(), "duplicate pairs in batch")
                # print(identical_nondup.shape, identical_nondup.sum().item(), "non-duplicate pairs in batch")

                # x_left = x_left[~identical_nondup]
                # x_right = x_right[~identical_nondup]
                # y = y[~identical_nondup]

                e_l = self.trainer.embed_t5(x_left)
                e_r = self.trainer.embed_t5(x_right)
                d   = torch.sqrt(((e_l - e_r) ** 2).sum(dim=1, keepdim=True) + 1e-6)

                phi_left  = np.arctan2(x_left[:, 2], x_left[:, 1])
                phi_right = np.arctan2(x_right[:, 2], x_right[:, 1])
                eta_left  = x_left[:, 0]  * ETA_MAX     # undo η‑normalisation
                eta_right = x_right[:, 0] * ETA_MAX
                dphi = (phi_left - phi_right + np.pi) % (2*np.pi) - np.pi
                deta = eta_left - eta_right
                dR = (dphi**2 + deta**2)**0.5


                dist_t5_all.append(d.cpu().numpy().flatten())
                lbl_t5_all .append(y.numpy().flatten())
                dr_t5_all  .append(dR)


        dist_t5_all = np.concatenate(dist_t5_all)
        lbl_t5_all  = np.concatenate(lbl_t5_all)
        dr_t5_all   = np.concatenate(dr_t5_all)

        print(f"T5-T5 pairs:   {len(dist_t5_all)} distances")
        print(f"  Range: min={dist_t5_all.min():.4f}, max={dist_t5_all.max():.4f}")
        print(f"  Labels: similar={(lbl_t5_all==0).sum()}, dissimilar={(lbl_t5_all==1).sum()}")

        # 2) histogram of distances
        fig, ax = plt.subplots(figsize=(10,6))
        # plt.figure(figsize=(10,6))
        bins = np.linspace(0, 7.5, 100)
        ax.hist(dist_t5_all[lbl_t5_all==0], bins=bins, alpha=0.6, label='Duplicates (0)')
        ax.hist(dist_t5_all[lbl_t5_all==1], bins=bins, alpha=0.6, label='Non-Duplicates (1)')
        ax.semilogy()
        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Number of pairs")
        ax.set_title("T5-T5 Embedding Distance Distribution (Test Set)")
        ax.tick_params(top=True, right=True)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0,7.5)
        pdf.savefig(fig)
        plt.close(fig)
        # plt.show()

        for xmax in [0.8, 0.2]:
            fig, ax = plt.subplots(figsize=(10,6))
            # plt.figure(figsize=(10,6))
            bins = np.linspace(0, xmax, 100)
            ax.hist(dist_t5_all[lbl_t5_all==0], bins=bins, alpha=0.6, label='Duplicates (0)')
            ax.hist(dist_t5_all[lbl_t5_all==1], bins=bins, alpha=0.6, label='Non-Duplicates (1)')
            ax.semilogy()
            ax.set_xlabel("Euclidean Distance")
            ax.set_ylabel("Number of pairs")
            ax.set_title("T5-T5 Embedding Distance Distribution (Test Set)")
            ax.legend()
            ax.tick_params(top=True, right=True)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, xmax)
            pdf.savefig(fig)
            plt.close(fig)
            # plt.show()


        # 2) histogram of dR
        fig, ax = plt.subplots(figsize=(10,6))
        bins = np.linspace(0, 0.15, 100)
        ax.hist(dr_t5_all[lbl_t5_all==0], bins=bins, alpha=0.6, label='Duplicates (0)')
        ax.hist(dr_t5_all[lbl_t5_all==1], bins=bins, alpha=0.6, label='Non-Duplicates (1)')
        ax.semilogy()
        ax.set_xlabel("Delta R")
        ax.set_ylabel("Number of pairs")
        ax.set_title("T5-T5 Embedding Distance Distribution (Test Set)")
        ax.tick_params(top=True, right=True)
        ax.legend()
        ax.grid(alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)
        # plt.show()

        # 2D histogram of dR vs distance
        fig, ax = plt.subplots(figsize=(10,6))
        _, _, _, im = ax.hist2d(dist_t5_all[lbl_t5_all==1],
                                dr_t5_all[lbl_t5_all==1],
                                bins=[np.linspace(0, 7.5, 100),
                                      np.linspace(0, 0.14, 100)],
                                cmap='viridis',
                                cmin=0.5)
        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Delta R")
        ax.set_title("2D Histogram of T5-T5 Distances (non-duplicates)")
        ax.tick_params(top=True, right=True)
        fig.colorbar(im, ax=ax, label="Number of pairs")

        # linear scale
        pdf.savefig(fig)
        plt.close(fig)

        # log scale
        xmaxs = [7.5, 2.0, 0.3]
        for xmax in xmaxs:
            fig, ax = plt.subplots(figsize=(10,6))
            _, _, _, im = ax.hist2d(dist_t5_all[lbl_t5_all==1],
                                    dr_t5_all[lbl_t5_all==1],
                                    norm=mpl.colors.LogNorm(),
                                    bins=[np.linspace(0, xmax, 100),
                                          np.linspace(0, 0.14, 100)],
                                    cmap='viridis',
                                    cmin=0.5)
            ax.set_xlabel("Euclidean Distance")
            ax.set_ylabel("Delta R")
            ax.set_title("2D Histogram of T5-T5 Distances (non-duplicates)")
            ax.tick_params(top=True, right=True)
            fig.colorbar(im, ax=ax, label="Number of pairs")
            pdf.savefig(fig)
            plt.close(fig)


        # ------------------------------------------------------------------
        # ΔR² baseline for the same test split
        # ------------------------------------------------------------------
        phi_left  = np.arctan2(self.trainer.X_left_test[:, 2], self.trainer.X_left_test[:, 1])
        phi_right = np.arctan2(self.trainer.X_right_test[:, 2], self.trainer.X_right_test[:, 1])
        eta_left  = self.trainer.X_left_test[:, 0]  * ETA_MAX     # undo η‑normalisation
        eta_right = self.trainer.X_right_test[:, 0] * ETA_MAX

        dphi = (phi_left - phi_right + np.pi) % (2*np.pi) - np.pi
        deta = eta_left - eta_right
        dRsq = dphi**2 + deta**2

        # 3) ROC curves: embedding vs ΔR² baseline
        fpr_t5, tpr_t5, _ = roc_curve(lbl_t5_all, -dist_t5_all, pos_label=0)
        fpr_dr, tpr_dr, _ = roc_curve(lbl_t5_all, -dRsq,        pos_label=0)

        auc_t5 = auc(fpr_t5, tpr_t5)
        auc_dr = auc(fpr_dr,  tpr_dr)

        fig, ax = plt.subplots(figsize=(7,7))
        # plt.figure(figsize=(7,7))
        ax.plot(fpr_t5, tpr_t5, label=f"Embedding distance (AUC={auc_t5:.3f})")
        ax.plot(fpr_dr, tpr_dr, '--', label=f"ΔR² baseline (AUC={auc_dr:.3f})")
        ax.plot([0,1],[0,1], '--', color='grey')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("T5-T5 Duplicate Discrimination")
        ax.legend()
        ax.grid(alpha=0.3)
        # plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        ax.set_ylim([0.95, 1.0])
        pdf.savefig(fig)
        plt.close(fig)

        print(f"T5-T5 AUC (embedding) = {auc_t5:.4f}")
        print(f"T5-T5 AUC (ΔR²)       = {auc_dr :.4f}")



