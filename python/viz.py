import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from tqdm import tqdm
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams.update({"font.size": 16})


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ETA_MAX = 2.5
ETA_MAX_PLS = 4.0

class Plotter:

    def __init__(self, trainer):
        self.trainer = trainer


    def plot(self, pdf_path: Path):
        with PdfPages(pdf_path) as pdf:
            self.plot_t5t5_performance(pdf)
            self.plot_t5pls_performance(pdf)
            self.plot_loss_per_epoch(pdf)
            self.plot_phi_comparison(pdf)
            self.plot_features_of_pairs(pdf)


    def plot_loss_per_epoch(self, pdf: PdfPages):
        if not hasattr(self.trainer, "losses_t5t5") or not hasattr(self.trainer, "losses_t5pls"):
            print("No losses to plot. Make sure to call train() first.")
            return
        # Plot training losses
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(self.trainer.losses_t5t5, label="T5-T5 Loss")
        ax.plot(self.trainer.losses_t5pls, label="T5-PLS Loss")
        ax.plot(np.array(self.trainer.losses_t5t5) + np.array(self.trainer.losses_t5pls), label="Total Loss", color="black")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Losses")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(top=True, right=True, which="both")
        ax.set_ylim(0.0, 0.5)
        pdf.savefig(fig)
        plt.close(fig)

        ax.set_ylim(1e-2, 1e0)
        ax.semilogy()
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

    def plot_t5pls_performance(self, pdf: PdfPages):

        self.trainer.embed_pls.eval(); self.trainer.embed_t5.eval()

        # 1) collect distances and labels ------------------------------------
        dist_pls_all = []
        lbl_pls_all  = []

        with torch.no_grad():
            for pls_feats, t5_feats, y, _ in self.trainer.test_pls_loader:
                # pls_feats = pls_feats.to(device)
                # t5_feats  = t5_feats.to(device)

                e_p = self.trainer.embed_pls(pls_feats)
                e_t = self.trainer.embed_t5(t5_feats)
                d   = torch.sqrt(((e_p - e_t) ** 2).sum(dim=1, keepdim=True) + 1e-6)

                dist_pls_all.append(d.cpu().numpy().flatten())
                lbl_pls_all .append(y.numpy().flatten())

        dist_pls_all = np.concatenate(dist_pls_all)
        lbl_pls_all  = np.concatenate(lbl_pls_all)

        print(f"pLS-T5 pairs:  {len(dist_pls_all)} distances")
        print(f"  Range: min={dist_pls_all.min():.4f}, max={dist_pls_all.max():.4f}")
        print(f"  Labels: similar={(lbl_pls_all==0).sum()}, dissimilar={(lbl_pls_all==1).sum()}")


        # 2) histogram of distances ------------------------------------------
        plt.figure(figsize=(10,6))
        bins = np.linspace(0, dist_pls_all.max(), 100)
        plt.hist(dist_pls_all[lbl_pls_all==0], bins=bins, density=True, alpha=0.6, label='Duplicates (0)')
        plt.hist(dist_pls_all[lbl_pls_all==1], bins=bins, density=True, alpha=0.6, label='Non-Duplicates (1)')
        plt.yscale('log')
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Density")
        plt.title("pLS-T5 Embedding Distance Distribution (Test Set)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(left=0)
        # plt.show()
        pdf.savefig()
        plt.close()

        # --------------------------------------------------------------------
        # ΔR² baseline for the same pLS-T5 pairs
        # --------------------------------------------------------------------
        #  — recover φ and η for the raw feature arrays used in the split
        phi_left  = np.arctan2(self.trainer.X_pls_test[:, 3],  self.trainer.X_pls_test[:, 2])     # pLS: sinφ, cosφ
        phi_right = np.arctan2(self.trainer.X_t5raw_test[:, 2], self.trainer.X_t5raw_test[:, 1])  # T5 : sinφ, cosφ
        eta_left  = self.trainer.X_pls_test[:, 0]  * 4.0        # pLS stored η/4
        eta_right = self.trainer.X_t5raw_test[:, 0] * ETA_MAX   # T5  stored η/η_max

        dphi = (phi_left - phi_right + np.pi) % (2*np.pi) - np.pi
        deta = eta_left - eta_right
        dRsq = dphi**2 + deta**2

        # 3) ROC curves: embedding vs ΔR² baseline ---------------------------
        fpr_pls, tpr_pls, _ = roc_curve(lbl_pls_all, -dist_pls_all, pos_label=0)
        fpr_dr,  tpr_dr,  _ = roc_curve(lbl_pls_all, -dRsq,         pos_label=0)

        auc_pls = auc(fpr_pls, tpr_pls)
        auc_dr  = auc(fpr_dr,  tpr_dr)

        plt.figure(figsize=(7,7))
        plt.plot(fpr_pls, tpr_pls, label=f"Embedding distance (AUC={auc_pls:.3f})")
        plt.plot(fpr_dr,  tpr_dr,  '--', label=f"ΔR² baseline (AUC={auc_dr:.3f})")
        plt.plot([0,1],[0,1], '--', color='grey')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("pLS-T5 Duplicate Discrimination")
        plt.legend()
        plt.grid(alpha=0.3)
        # plt.show()
        pdf.savefig()
        plt.close()

        print(f"pLS-T5 AUC (embedding) = {auc_pls:.4f}")
        print(f"pLS-T5 AUC (ΔR²)       = {auc_dr :.4f}")


    def plot_phi_comparison(self, pdf: PdfPages):
        print("Plotting phi comparison")
        print(self.trainer.X_pls_test.shape)
        print(self.trainer.X_t5raw_test.shape)
        mask = self.trainer.y_pls_test == 0

        # first plot: assume we're using cos, sin
        t5_phi = np.arctan2(self.trainer.X_t5raw_test[:, 2], self.trainer.X_t5raw_test[:, 1])
        pls_phi = np.arctan2(self.trainer.X_pls_test[:, 3], self.trainer.X_pls_test[:, 2])
        pls_pt_inv = self.trainer.X_pls_test[:, 4]
        dphi = pls_phi - t5_phi
        dphi[dphi > np.pi] -= 2 * np.pi
        dphi[dphi < -np.pi] += 2 * np.pi

        # plot
        fig, ax = plt.subplots(figsize=(8, 8))
        _, _, _, im = ax.hist2d(pls_pt_inv[mask], dphi[mask], bins=100, cmin=0.5, cmap='rainbow')
        ax.set_xlabel("pLS 1/pT [GeV]")
        ax.set_ylabel("Duplicate pLS phi - T5 phi")
        ax.set_title("On the assumption features are cos(phi), sin(phi)")
        fig.colorbar(im, ax=ax, label="Pairs")
        fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.94)
        pdf.savefig()
        plt.close()

        # second plot: assume we're using phi, phi+pi
        t5_phi = self.trainer.X_t5raw_test[:, 1]
        pls_phi = self.trainer.X_pls_test[:, 2]
        pls_pt_inv = self.trainer.X_pls_test[:, 4]
        dphi = pls_phi - t5_phi
        dphi[dphi > np.pi] -= 2 * np.pi
        dphi[dphi < -np.pi] += 2 * np.pi

        # plot
        fig, ax = plt.subplots(figsize=(8, 8))
        _, _, _, im = ax.hist2d(pls_pt_inv[mask], dphi[mask], bins=100, cmin=0.5, cmap='rainbow')
        ax.set_xlabel("pLS 1/pT [GeV]")
        ax.set_ylabel("Duplicate pLS phi - T5 phi")
        ax.set_title("On the assumption features are phi, phi+pi")
        fig.colorbar(im, ax=ax, label="Pairs")
        fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.94)
        pdf.savefig()
        plt.close()


    def plot_features_of_pairs(self, pdf: PdfPages):
        print("Plotting features of pairs")

        n_features, x_test = {}, {}

        mask = self.trainer.y_pls_test == 0
        _, n_features["pls"] = self.trainer.X_pls_test.shape
        _, n_features["t5"] = self.trainer.X_t5raw_test.shape
        x_test["pls"] = self.trainer.X_pls_test[mask]
        x_test["t5"] = self.trainer.X_t5raw_test[mask]

        for track in ["pls", "t5"]:
            print(f"Track: {track}")
            print(f"Number of features: {n_features[track]}")
            print(f"Test data shape: {x_test[track].shape}")
            for feature in range(n_features[track]):
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.hist(x_test[track][:, feature], bins=100)
                ax.set_xlabel(f"{track} feature {feature}")
                ax.set_ylabel("Tracks")
                ax.grid(alpha=0.3)
                pdf.savefig()
                plt.close()



class PlotterPtEtaPhi:


    def __init__(self,
                 trainer,
                 train_emb,
                 use_dxy_dz: bool,
                 ):
        self.trainer = trainer
        self.train_emb = train_emb
        self.use_dxy_dz = use_dxy_dz
        self.columns = ["qoverpt", "eta", "cosphi", "sinphi", "pca_dxy", "pca_dz"]
        if not self.use_dxy_dz:
            self.columns = self.columns[:-2]
        self.lw = 3
        self.embed_test_data()
        self.make_dataframes()


    def embed_test_data(self):
        print("Embedding test data")
        self.trainer.embed_t5.eval()
        self.trainer.embed_pls.eval()
        with torch.no_grad():
            self.emb_t5_test = self.trainer.embed_t5(torch.tensor(self.trainer.features_t5_test))
            self.emb_pls_test = self.trainer.embed_pls(torch.tensor(self.trainer.features_pls_test))
        print(f"Embed T5 test shape: {self.emb_t5_test.shape}")
        print(f"Embed PLS test shape: {self.emb_pls_test.shape}")


    def make_dataframes(self):
        self.make_dataframes_features()
        self.make_dataframes_sim_features()
        self.make_dataframes_embeddings()


    def make_dataframes_features(self):
        pass


    def make_dataframes_sim_features(self):
        self.df_sim_t5 = pd.DataFrame(self.trainer.sim_features_t5_test, columns=self.columns)
        self.df_sim_pls = pd.DataFrame(self.trainer.sim_features_pls_test, columns=self.columns)
        self.df_sim_t5["phi"] = np.arctan2(self.df_sim_t5["sinphi"], self.df_sim_t5["cosphi"])
        self.df_sim_pls["phi"] = np.arctan2(self.df_sim_pls["sinphi"], self.df_sim_pls["cosphi"])

    def make_dataframes_embeddings(self):
        self.df_emb_t5 = pd.DataFrame(self.emb_t5_test.numpy(), columns=self.columns)
        self.df_emb_pls = pd.DataFrame(self.emb_pls_test.numpy(), columns=self.columns)
        self.df_emb_t5["phi"] = np.arctan2(self.df_emb_t5["sinphi"], self.df_emb_t5["cosphi"])
        self.df_emb_pls["phi"] = np.arctan2(self.df_emb_pls["sinphi"], self.df_emb_pls["cosphi"])


    def plot(self, pdf_path: Path):
        self.cmap = "plasma"
        with PdfPages(pdf_path) as pdf:
            self.plot_roc_curves(pdf)
            self.plot_performance(pdf)


    def plot_roc_curves(self, pdf: PdfPages):
        if self.train_emb is None:
            return
        self.plot_t5t5_roc_curves(pdf)
        self.plot_t5pls_roc_curves(pdf)


    def plot_t5t5_roc_curves(self, pdf: PdfPages):
        print("Plotting T5-T5 ROC curves")
        dist_t5_all = []
        dphys_t5_all = []
        dr_t5_all   = []
        lbl_t5_all  = []

        with torch.no_grad():
            for x_left, x_right, y, _ in tqdm(self.train_emb.test_t5_loader):

                # demb
                e_l = self.train_emb.embed_t5(x_left)
                e_r = self.train_emb.embed_t5(x_right)
                d   = torch.sqrt(((e_l - e_r) ** 2).sum(dim=1, keepdim=True) + 1e-6)

                # dR
                phi_left  = np.arctan2(x_left[:, 2], x_left[:, 1])
                phi_right = np.arctan2(x_right[:, 2], x_right[:, 1])
                eta_left  = x_left[:, 0]  * ETA_MAX     # undo η‑normalisation
                eta_right = x_right[:, 0] * ETA_MAX
                dphi = (phi_left - phi_right + np.pi) % (2*np.pi) - np.pi
                deta = eta_left - eta_right
                dR = (dphi**2 + deta**2)**0.5

                # dphys
                phys_l = self.trainer.embed_t5(x_left)
                phys_r = self.trainer.embed_t5(x_right)
                dphys = torch.sqrt(((phys_l - phys_r) ** 2).sum(dim=1, keepdim=True) + 1e-6)

                # store them
                dist_t5_all.append(d.cpu().numpy().flatten())
                dphys_t5_all.append(dphys.cpu().numpy().flatten())
                lbl_t5_all.append(y.numpy().flatten())
                dr_t5_all.append(dR)


        dist_t5_all = np.concatenate(dist_t5_all)
        dphys_t5_all = np.concatenate(dphys_t5_all)
        lbl_t5_all  = np.concatenate(lbl_t5_all)
        dr_t5_all   = np.concatenate(dr_t5_all)

        # ROC curves
        fpr_emb, tpr_emb, _ = roc_curve(lbl_t5_all, -dist_t5_all, pos_label=0)
        fpr_dr, tpr_dr, _ = roc_curve(lbl_t5_all, -dr_t5_all,  pos_label=0)
        fpr_dphys, tpr_dphys, _ = roc_curve(lbl_t5_all, -dphys_t5_all,  pos_label=0)

        auc_t5 = auc(fpr_emb, tpr_emb)
        auc_dr = auc(fpr_dr,  tpr_dr)
        auc_dphys = auc(fpr_dphys,  tpr_dphys)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr_emb, tpr_emb, linewidth=self.lw, label=f"Embedding distance, AUC={auc_t5:.3f}")
        ax.plot(fpr_dr, tpr_dr, '--', linewidth=self.lw, label=f"ΔR² baseline, AUC={auc_dr:.3f}")
        ax.plot(fpr_dphys, tpr_dphys, ':', linewidth=self.lw, label=f"(q/pt, eta, phi) distance, AUC={auc_dphys:.3f}")
        ax.plot([0,1],[0,1], '--', linewidth=self.lw, color='grey')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("T5-T5 Duplicate Discrimination")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        pdf.savefig(fig)
        plt.close(fig)


    def plot_t5pls_roc_curves(self, pdf: PdfPages):
        print("Plotting T5-PLS ROC curves")
        dist_pls_all = []
        dphys_pls_all = []
        lbl_pls_all = []
        dr_pls_all = []

        with torch.no_grad():
            for pls_feats, t5_feats, y, _ in tqdm(self.train_emb.test_pls_loader):

                # demb
                e_p = self.train_emb.embed_pls(pls_feats)
                e_t = self.train_emb.embed_t5(t5_feats)
                d   = torch.sqrt(((e_p - e_t) ** 2).sum(dim=1, keepdim=True) + 1e-6)

                # dR
                phi_t5  = np.arctan2(t5_feats[:, 2], t5_feats[:, 1])
                phi_pls = np.arctan2(pls_feats[:, 3], pls_feats[:, 2])
                eta_t5  = t5_feats[:, 0]  * ETA_MAX     # undo η‑normalisation
                eta_pls = pls_feats[:, 0] * ETA_MAX_PLS
                dphi = (phi_t5 - phi_pls + np.pi) % (2*np.pi) - np.pi
                deta = eta_t5 - eta_pls
                dR = (dphi**2 + deta**2)**0.5

                # dphys
                phys_p = self.trainer.embed_pls(pls_feats)
                phys_t = self.trainer.embed_t5(t5_feats)
                dphys = torch.sqrt(((phys_p - phys_t) ** 2).sum(dim=1, keepdim=True) + 1e-6)

                dist_pls_all.append(d.cpu().numpy().flatten())
                lbl_pls_all.append(y.numpy().flatten())
                dphys_pls_all.append(dphys.cpu().numpy().flatten())
                dr_pls_all.append(dR)

        dist_pls_all = np.concatenate(dist_pls_all)
        lbl_pls_all = np.concatenate(lbl_pls_all)
        dphys_pls_all = np.concatenate(dphys_pls_all)
        dr_pls_all = np.concatenate(dr_pls_all)

        # ROC curves
        fpr_emb, tpr_emb, _ = roc_curve(lbl_pls_all, -dist_pls_all, pos_label=0)
        fpr_dr, tpr_dr, _ = roc_curve(lbl_pls_all, -dr_pls_all, pos_label=0)
        fpr_dphys, tpr_dphys, _ = roc_curve(lbl_pls_all, -dphys_pls_all, pos_label=0)

        auc_t5 = auc(fpr_emb, tpr_emb)
        auc_dr = auc(fpr_dr,  tpr_dr)
        auc_dphys = auc(fpr_dphys,  tpr_dphys)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr_emb, tpr_emb, linewidth=self.lw, label=f"Embedding distance, AUC={auc_t5:.3f}")
        ax.plot(fpr_dr, tpr_dr, '--', linewidth=self.lw, label=f"ΔR² baseline, AUC={auc_dr:.3f}")
        ax.plot(fpr_dphys, tpr_dphys, ':', linewidth=self.lw, label=f"(q/pt, eta, phi) distance, AUC={auc_dphys:.3f}")
        ax.plot([0,1],[0,1], '--', linewidth=self.lw, color='grey')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("T5-pLS Duplicate Discrimination")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        pdf.savefig(fig)
        plt.close(fig)


    def plot_performance(self, pdf: PdfPages):

        bins = np.linspace(-1.0, 1.0, 100)

        for (df_sim, df_emb, track) in [
            (self.df_sim_t5, self.df_emb_t5, "T5"),
            (self.df_sim_pls, self.df_emb_pls, "PLS")
        ]:

            # deta
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(df_sim["eta"] - df_emb["eta"], bins=bins)
            ax.set_xlabel(f"Sim. eta - Predicted {track} eta")
            ax.set_ylabel("Tracks")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(top=True, right=True, direction="in")
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.96)
            pdf.savefig()
            ax.semilogy()
            pdf.savefig()
            plt.close()

            # deta vs eta
            fig, ax = plt.subplots(figsize=(8, 8))
            _, _, _, im = ax.hist2d(
                df_sim["eta"],
                df_sim["eta"] - df_emb["eta"],
                bins=(100, bins),
                cmin=0.5,
                cmap=self.cmap,
                )
            fig.colorbar(im, ax=ax, label="Tracks")
            ax.set_xlabel("Sim. eta")
            ax.set_ylabel(f"Sim. eta - Predicted {track} eta")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(top=True, right=True, direction="in")
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.96)
            pdf.savefig()
            plt.close()

            # dphi
            dphi = df_sim["phi"] - df_emb["phi"]
            dphi[dphi > np.pi] -= 2 * np.pi
            dphi[dphi < -np.pi] += 2 * np.pi
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(dphi, bins=bins)
            ax.set_xlabel(f"Sim. phi - Predicted {track} phi")
            ax.set_ylabel("Tracks")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(top=True, right=True, direction="in")
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.96)
            pdf.savefig()
            ax.semilogy()
            pdf.savefig()
            plt.close()

            # dphi vs phi
            fig, ax = plt.subplots(figsize=(8, 8))
            _, _, _, im = ax.hist2d(
                df_sim["phi"],
                dphi,
                bins=(100, bins),
                cmin=0.5,
                cmap=self.cmap,
                )
            fig.colorbar(im, ax=ax, label="Tracks")
            ax.set_xlabel("Sim. phi")
            ax.set_ylabel(f"Sim. phi - Predicted {track} phi")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(top=True, right=True, direction="in")
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.96)
            pdf.savefig()
            plt.close()

            # dphi vs q/pt
            fig, ax = plt.subplots(figsize=(8, 8))
            _, _, _, im = ax.hist2d(
                df_sim["qoverpt"],
                dphi,
                bins=(100, bins),
                cmin=0.5,
                cmap=self.cmap,
                )
            fig.colorbar(im, ax=ax, label="Tracks")
            ax.set_xlabel("Sim. q/pt")
            ax.set_ylabel(f"Sim. phi - Predicted {track} phi")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(top=True, right=True, direction="in")
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.96)
            pdf.savefig()
            plt.close()

            # dq/pt
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(df_sim["qoverpt"] - df_emb["qoverpt"], bins=bins)
            ax.set_xlabel(f"Sim. q/pt - Predicted {track} q/pt")
            ax.set_ylabel("Tracks")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(top=True, right=True, direction="in")
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.96)
            pdf.savefig()
            ax.semilogy()
            pdf.savefig()
            plt.close()

            # dq/pt vs q/pt
            fig, ax = plt.subplots(figsize=(8, 8))
            _, _, _, im = ax.hist2d(
                df_sim["qoverpt"],
                df_sim["qoverpt"] - df_emb["qoverpt"],
                bins=(100, bins),
                cmin=0.5,
                cmap=self.cmap,
                )
            fig.colorbar(im, ax=ax, label="Tracks")
            ax.set_xlabel("Sim. q/pt")
            ax.set_ylabel(f"Sim. q/pt - Predicted {track} q/pt")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(top=True, right=True, direction="in")
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.93, top=0.96)
            pdf.savefig()
            plt.close()
