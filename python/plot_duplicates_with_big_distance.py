import awkward as ak
import numpy as np
from pathlib import Path
import pickle
import uproot
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

# from preprocess import load_root_file
from preprocess import branches_list
from ml import EmbeddingNetT5, EmbeddingNetpLS

PAIRS_T5T5 = Path("pairs_t5t5.pkl")
PAIRS_T5PLS = Path("pairs_t5pls.pkl")
BRANCHES = Path("branches.pkl")
BONUS_FEATURES = 2
SIM_BRANCHES = [
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_pdgId",
]
TRACKING_NTUPLE = Path("/ceph/cms/store/user/evourlio/LST/samples/CMSSW_12_2_0_pre2/RelValTTbar_14TeV_CMSSW_12_5_0_pre3/event_1000.root")


def main():
    network = Path("model_weights.pth")
    file_path = Path("/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75.root")
    # file_path = Path("/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p50.root")

    plotter = PlotDuplicatesWithBigDistance()
    plotter.load_network(network)
    plotter.load_t5_t5_pairs()
    plotter.load_t5_pls_pairs()
    plotter.load_test_scores()
    # plotter.load_tracking_ntuple(TRACKING_NTUPLE, SIM_BRANCHES)
    # plotter.load_raw_data(file_path)
    # pairs = plotter.get_pls_t5_pairs_of_interest()
    # plotter.plot_t5_pls_pairs(pairs)

    with PdfPages("duplicates_with_big_distance.pdf") as pdf:
        plotter.plot_t5_t5_hists(pdf)
        plotter.plot_t5_pls_hists(pdf)

    # pairs_t5t5 = plotter.get_t5_t5_pairs_of_interest()
    # plotter.plot_t5_t5_pairs(pairs_t5t5)




class PlotDuplicatesWithBigDistance:
    def __init__(self):
        self.embed_t5 = EmbeddingNetT5()
        self.embed_pls = EmbeddingNetpLS()


    def load_network(self, path: Path):
        print(f"Loading model from {path} ...")
        checkpoint = torch.load(path)
        self.embed_t5.load_state_dict(checkpoint["embed_t5"])
        self.embed_pls.load_state_dict(checkpoint["embed_pls"])
        self.embed_t5.eval()
        self.embed_pls.eval()


    def load_test_scores(self):
        self.load_t5_t5_test_scores()
        self.load_pls_t5_test_scores()


    def load_t5_t5_test_scores(self):
        print("Calculating T5-T5 scores ...")
        with torch.no_grad():
            x_left = torch.tensor(self.X_left_test[:, :-BONUS_FEATURES])
            x_right = torch.tensor(self.X_right_test[:, :-BONUS_FEATURES])
            self.e_l = self.embed_t5(x_left).detach().numpy()
            self.e_r = self.embed_t5(x_right).detach().numpy()
            self.d_t5t5 = np.sqrt(((self.e_l - self.e_r) ** 2).sum(axis=1, keepdims=True) + 1e-6)
        self.d_t5t5 = self.d_t5t5.flatten()


    def load_pls_t5_test_scores(self):
        print("Calculating PLS-T5 scores ...")
        with torch.no_grad():
            x_pls = torch.tensor(self.X_pls_test[:, :-BONUS_FEATURES])
            x_t5 = torch.tensor(self.X_t5raw_test[:, :-BONUS_FEATURES])
            self.e_pls = self.embed_pls(x_pls).detach().numpy()
            self.e_t5 = self.embed_t5(x_t5).detach().numpy()
            self.d_plst5 = np.sqrt(((self.e_pls - self.e_t5) ** 2).sum(axis=1, keepdims=True) + 1e-6)
        self.d_plst5 = self.d_plst5.flatten()


    def load_raw_data(self, file_path):
        print(f"Loading raw data from {file_path} ...")
        self.data = load_root_file_ak(file_path, branches_list + SIM_BRANCHES)


    def load_tracking_ntuple(self, file_path, branches):
        print(f"Loading tracking ntuple from {file_path} ...")
        self.sim = load_tracking_ntuple(file_path, branches)


    def load_t5_t5_pairs(self):
        print("Getting T5-T5 pairs")
        [self.X_left_train,
         self.X_left_test,
         self.X_right_train,
         self.X_right_test,
         self.y_t5_train,
         self.y_t5_test,
         self.w_t5_train,
         self.w_t5_test,
         self.true_L_train,
         self.true_L_test,
         self.true_R_train,
         self.true_R_test
         ] = load_t5_t5_pairs()


    def load_t5_pls_pairs(self):
        print("Getting PLS-T5 pairs")
        [self.X_pls_train,
         self.X_pls_test,
         self.X_t5raw_train,
         self.X_t5raw_test,
         self.y_pls_train,
         self.y_pls_test,
         self.w_pls_train,
         self.w_pls_test,
         ] = load_t5_pls_pairs()


    def get_t5_features(self):
        pass


    def get_t5_t5_pairs_of_interest(self, max_pairs=1):
        dup_mask = self.y_t5_test == 0
        dup_dist = self.d_t5t5[dup_mask]
        dup_x_l = self.X_left_test[dup_mask]
        dup_x_r = self.X_right_test[dup_mask]
        dup_idxs = np.flip(np.argsort(dup_dist))
        ret = []
        for i in range(max_pairs):
            idx = dup_idxs[i]
            ev_l, i_l = dup_x_l[idx][-2], dup_x_l[idx][-1]
            ev_r, i_r = dup_x_r[idx][-2], dup_x_r[idx][-1]
            if ev_l != ev_r:
                raise ValueError(f"Event mismatch: {ev_l} != {ev_r}")
            ret.append( (int(ev_l), int(i_l), int(i_r), idx, dup_dist[idx]) )
        return ret


    def get_pls_t5_pairs_of_interest(self, max_pairs=5):
        # Get pairs of PLS-T5 duplicates with the largest distance
        dup_mask = self.y_pls_test == 0
        dup_dist = self.d_plst5[dup_mask]
        dup_x_t5 = self.X_t5raw_test[dup_mask]
        dup_x_pls = self.X_pls_test[dup_mask]
        dup_idxs = np.flip(np.argsort(dup_dist))
        ret = []
        for i in range(max_pairs):
            idx = dup_idxs[i]
            ev_pls, i_pls = dup_x_pls[idx][-2], dup_x_pls[idx][-1]
            ev_t5, i_t5   = dup_x_t5[idx][-2], dup_x_t5[idx][-1]
            if ev_pls != ev_t5:
                raise ValueError(f"Event mismatch: {ev_pls} != {ev_t5}")
            ret.append( (int(ev_pls), int(i_pls), int(i_t5), dup_dist[idx]) )
        return ret
        # return [
        #     (199, 2_588, 536),
        # ]


    def plot_t5_t5_pairs(self, pairs):
        dup_mask = self.y_t5_test == 0
        dup_x_l = self.X_left_test[dup_mask]
        dup_x_r = self.X_right_test[dup_mask]

        dif_lr = dup_x_l[:, :-BONUS_FEATURES] - dup_x_r[:, :-BONUS_FEATURES]
        std_lr = np.std(dif_lr, axis=0)
        print(f"Std dev of differences between left and right features: {std_lr.shape}")
        print(std_lr)

        for (ev, i_l, i_r, idx, dist) in pairs:
            eta_l, phi_l = self.data["t5_eta"][ev][i_l], self.data["t5_phi"][ev][i_l]
            eta_r, phi_r = self.data["t5_eta"][ev][i_r], self.data["t5_phi"][ev][i_r]
            simidx_l = self.data["t5_matched_simIdx"][ev][i_l]
            simidx_r = self.data["t5_matched_simIdx"][ev][i_r]
            pmatched_l = self.data["t5_pMatched"][ev][i_l]
            pmatched_r = self.data["t5_pMatched"][ev][i_r]
            deltar2 = (eta_l - eta_r) ** 2 + (phi_l - phi_r) ** 2
            if simidx_l != simidx_r or len(simidx_l) != 1:
                print("WARNING: simidx are weird!")
            simidx = simidx_l[0]
            sim_pt, sim_eta, sim_phi, sim_pdgId = (
                self.sim["sim_pt"][ev][simidx],
                self.sim["sim_eta"][ev][simidx],
                self.sim["sim_phi"][ev][simidx],
                self.sim["sim_pdgId"][ev][simidx],
            )
            feat_l = dup_x_l[idx][:-BONUS_FEATURES]
            feat_r = dup_x_r[idx][:-BONUS_FEATURES]
            diff = feat_l - feat_r
            diff_normed = diff / std_lr
            xys_l = self.get_xys(ev, i_l)
            xys_r = self.get_xys(ev, i_r)
            nequal = sum([1 for xy_l, xy_r in zip(xys_l, xys_r) if xy_l == xy_r])
            print("")
            print(f"Event: {ev}")
            print(f"L index: {i_l}, R index: {i_r}")
            print(eta_l, phi_l)
            print(eta_r, phi_r)
            print(f"simIdx l: {simidx_l}")
            print(f"simIdx r: {simidx_r}")
            print(f"pMatched l: {pmatched_l}")
            print(f"pMatched r: {pmatched_r}")
            print(f"Delta R2: {deltar2:.3f}")
            print(f"Delta R: {np.sqrt(deltar2):.3f}")
            print(f"Distance: {dist:.3f}")
            print(f"Equal xys: {nequal} / {len(xys_l)}")
            print(f"Feat L: {feat_l}")
            print(f"Feat R: {feat_r}")
            print(f"Diff: {diff}")
            print(f"Normed diff: {diff_normed}")
            print(f"Sim index: {simidx} vs len(sim_pt): {len(self.sim['sim_pt'][ev])}")
            print(f"Sim: {sim_pdgId} pt={sim_pt:6.2f}, eta={sim_eta:6.3f}, phi={sim_phi:6.3f}")
            print("")


    def plot_t5_pls_pairs(self, pairs):
        for (ev, i_pls, i_t5, dist) in pairs:
            pls_eta, pls_phi = self.data["pLS_eta"][ev][i_pls], self.data["pLS_phi"][ev][i_pls]
            t5_eta, t5_phi = self.data["t5_eta"][ev][i_t5], self.data["t5_phi"][ev][i_t5]
            pls_simidx = self.data["pLS_matched_simIdx"][ev][i_pls]
            t5_simidx = self.data["t5_matched_simIdx"][ev][i_t5]
            t5_pmatched = self.data["t5_pMatched"][ev][i_t5]
            deltar2 = (pls_eta - t5_eta) ** 2 + (pls_phi - t5_phi) ** 2
            if pls_simidx != t5_simidx or len(pls_simidx) != 1:
                print("WARNING: simidx are weird!")
            simidx = pls_simidx[0]
            sim_pt, sim_eta, sim_phi, sim_pdgId = (
                self.sim["sim_pt"][ev][simidx],
                self.sim["sim_eta"][ev][simidx],
                self.sim["sim_phi"][ev][simidx],
                self.sim["sim_pdgId"][ev][simidx],
            )
            print("")
            print(f"Event: {ev}")
            print(f"pLS index: {i_pls}, T5 index: {i_t5}")
            print(pls_eta, pls_phi)
            print(t5_eta, t5_phi)
            print(f"pLS matched simIdx: {pls_simidx}")
            print(f"T5 matched simIdx: {t5_simidx}")
            print(f"T5 pMatched: {t5_pmatched}")
            print(f"Delta R2: {deltar2:.3f}")
            print(f"Delta R: {np.sqrt(deltar2):.3f}")
            print(f"Distance: {dist:.3f}")
            print(f"Sim index: {simidx} vs len(sim_pt): {len(self.sim['sim_pt'][ev])}")
            print(f"Sim: {sim_pdgId} pt={sim_pt:6.2f}, eta={sim_eta:6.3f}, phi={sim_phi:6.3f}")
            print("")


    def get_xys(self, ev, t5):
        triplets = [0, 1]
        layers = [0, 1, 2, 3, 4, 5]
        xys = []
        for triplet in triplets:
            for layer in layers:
                # Skip hits shared between triplets
                if triplet == 0 and layer in [4, 5]:
                    continue
                idx = self.data[f"t5_t3_idx{triplet}"][ev][t5]
                x = self.data[f"t5_t3_{layer}_x"][ev][idx]
                y = self.data[f"t5_t3_{layer}_y"][ev][idx]
                # print(f"T5 {t5}, triplet {triplet}, layer {layer}: x={x:.4f}, y={y:.4f}")
                xys.append((x, y))
        return xys


    def plot_t5_t5_hists(self, pdf: PdfPages):
        sim_mask = self.y_t5_test == 0

        # duplicates (similar)
        sim_x_l = self.X_left_test[sim_mask]
        sim_x_r = self.X_right_test[sim_mask]
        sim_e_l = self.e_l[sim_mask]
        sim_e_r = self.e_r[sim_mask]

        # non-duplicates (dissimilar)
        dis_x_l = self.X_left_test[~sim_mask]
        dis_x_r = self.X_right_test[~sim_mask]
        dis_e_l = self.e_l[~sim_mask]
        dis_e_r = self.e_r[~sim_mask]

        sim_dx_lr = sim_x_l[:, :-BONUS_FEATURES] - sim_x_r[:, :-BONUS_FEATURES]
        dis_dx_lr = dis_x_l[:, :-BONUS_FEATURES] - dis_x_r[:, :-BONUS_FEATURES]

        # 1D histograms of feature differences
        n_features = sim_dx_lr.shape[1]
        for feat in range(n_features):
            name = t5_feature_names(feat)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(sim_dx_lr[:, feat], bins=100, histtype="step", label=f"Feature {feat} ({name})", color="blue")
            ax.hist(dis_dx_lr[:, feat], bins=100, histtype="step", label=f"Feature {feat} ({name})", color="red")
            ax.set_xlabel(f"Difference between left and right feature {feat} ({name})")
            ax.set_ylabel("Count")
            ax.set_title(f"Feature {feat} differences between left and right T5 features")
            ax.tick_params(top=True, right=True, which="both", direction="in")
            ax.legend()
            pdf.savefig()
            plt.close()

        # 1D histograms of embedding differences
        sim_de_lr = sim_e_l - sim_e_r
        dis_de_lr = dis_e_l - dis_e_r
        n_emb = sim_e_l.shape[1]
        for emb in range(n_emb):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(sim_de_lr[:, emb], bins=100, histtype="step", label=f"Embedding {emb}", color="blue")
            ax.hist(dis_de_lr[:, emb], bins=100, histtype="step", label=f"Embedding {emb}", color="red")
            ax.set_xlabel(f"Difference between left and right embedding {emb}")
            ax.set_ylabel("Count")
            ax.set_title(f"Embedding {emb} differences between left and right T5 embeddings")
            ax.tick_params(top=True, right=True, which="both", direction="in")
            ax.legend()
            pdf.savefig()
            plt.close()

        # 2D histograms of feature differences and embedding differences
        fig_per_row = 3
        if n_emb % fig_per_row > 0:
            raise ValueError(f"Number of embeddings {n_emb} is not divisible by {fig_per_row}")

        for feat in range(n_features):

            name = t5_feature_names(feat)
            print(f"Plotting feature {feat} vs embeddings ...")
            #if feat > 1:
            #   break

            # duplicates (similar)
            fig, axs = plt.subplots(figsize=(16, 8), ncols=fig_per_row, nrows=n_emb // fig_per_row)
            for emb in range(n_emb):
                ax = axs[emb // fig_per_row, emb % fig_per_row]
                _, _, _, im = ax.hist2d(sim_dx_lr[:, feat], sim_de_lr[:, emb], bins=100, cmap="Blues", cmin=0.5, norm=colors.LogNorm())
                ax.set_xlabel(f"Feature {feat} difference")
                ax.set_ylabel(f"Embedding {emb} difference")
                ax.set_title(f"{name} vs Emb. {emb} (duplicates)")
                ax.tick_params(top=True, right=True, which="both", direction="in")
                ax.grid(alpha=0.3)
                fig.colorbar(im, ax=ax, label="Track pairs")
                fig.subplots_adjust(hspace=0.3, wspace=0.3, right=0.98, left=0.05, bottom=0.05, top=0.95)
            pdf.savefig()
            plt.close()

            # non-duplicates (dissimilar)
            fig, axs = plt.subplots(figsize=(16, 8), ncols=fig_per_row, nrows=n_emb // fig_per_row)
            for emb in range(n_emb):
                ax = axs[emb // fig_per_row, emb % fig_per_row]
                _, _, _, im = ax.hist2d(dis_dx_lr[:, feat], dis_de_lr[:, emb], bins=100, cmap="Reds", cmin=0.5, norm=colors.LogNorm())
                ax.set_xlabel(f"Feature {feat} difference")
                ax.set_ylabel(f"Embedding {emb} difference")
                ax.set_title(f"{name} vs Emb. {emb} (non-duplicates)")
                ax.tick_params(top=True, right=True, which="both", direction="in")
                ax.grid(alpha=0.3)
                fig.colorbar(im, ax=ax, label="Track pairs")
                fig.subplots_adjust(hspace=0.3, wspace=0.3, right=0.98, left=0.05, bottom=0.05, top=0.95)
                fig.subplots_adjust(hspace=0.3, wspace=0.3, right=0.98, left=0.05, bottom=0.05, top=0.95)
            pdf.savefig()
            plt.close()


    def plot_t5_pls_hists(self, pdf: PdfPages):
        sim_mask = self.y_pls_test == 0

        # duplicates (similar)
        sim_x_pls = self.X_pls_test[sim_mask][:, :-BONUS_FEATURES]
        sim_x_t5 = self.X_t5raw_test[sim_mask][:, :-BONUS_FEATURES]
        sim_e_pls = self.e_pls[sim_mask]
        sim_e_t5 = self.e_t5[sim_mask]
        sim_de = sim_e_pls - sim_e_t5

        # non-duplicates (dissimilar)
        dis_x_pls = self.X_pls_test[~sim_mask][:, :-BONUS_FEATURES]
        dis_x_t5 = self.X_t5raw_test[~sim_mask][:, :-BONUS_FEATURES]
        dis_e_pls = self.e_pls[~sim_mask]
        dis_e_t5 = self.e_t5[~sim_mask]
        dis_de = dis_e_pls - dis_e_t5

        # 1D histograms of features (t5)
        n_features_t5 = sim_x_t5.shape[1]
        for feat in range(n_features_t5):
            name = t5_feature_names(feat)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(sim_x_t5[:, feat], bins=100, histtype="step", label=f"Feature {feat} ({name})", color="blue")
            ax.hist(dis_x_t5[:, feat], bins=100, histtype="step", label=f"Feature {feat} ({name})", color="red")
            ax.set_xlabel(f"T5 Feature {feat} ({name})")
            ax.set_ylabel("Count")
            ax.set_title(f"T5 Feature {feat} ({name}) distribution")
            ax.tick_params(top=True, right=True, which="both", direction="in")
            ax.legend()
            pdf.savefig()
            plt.close()

        # 1D histograms of features (pls)
        n_features_pls = sim_x_pls.shape[1]
        for feat in range(n_features_pls):
            name = pls_feature_names(feat)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(sim_x_pls[:, feat], bins=100, histtype="step", label=f"Feature {feat} ({name})", color="blue")
            ax.hist(dis_x_pls[:, feat], bins=100, histtype="step", label=f"Feature {feat} ({name})", color="red")
            ax.set_xlabel(f"PLS Feature {feat} ({name})")
            ax.set_ylabel("Count")
            ax.set_title(f"PLS Feature {feat} ({name}) distribution")
            ax.tick_params(top=True, right=True, which="both", direction="in")
            ax.legend()
            pdf.savefig()
            plt.close()

        # 1D histograms of embedding differences
        sim_de_plst5 = sim_e_pls - sim_e_t5
        dis_de_plst5 = dis_e_pls - dis_e_t5
        n_emb = sim_e_pls.shape[1]
        for emb in range(n_emb):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(sim_de_plst5[:, emb], bins=100, histtype="step", label=f"Embedding {emb}", color="blue")
            ax.hist(dis_de_plst5[:, emb], bins=100, histtype="step", label=f"Embedding {emb}", color="red")
            ax.set_xlabel(f"Difference between left and right embedding {emb}")
            ax.set_ylabel("Count")
            ax.set_title(f"Embedding {emb} differences between left and right PLS and T5 embeddings")
            ax.tick_params(top=True, right=True, which="both", direction="in")
            ax.legend()
            pdf.savefig()
            plt.close()

        # a subset of 2D feature differences
        # (T5, PLS)
        feats = [
            (0, 0),
            (1, 2),
            (2, 3),
            (21, 9),
            (22, 9),
            (23, 9),
            (26, 7),
            (26, 8),
        ]

        # 2D histograms of feature differences and embedding differences
        fig_per_row = 3
        if n_emb % fig_per_row > 0:
            raise ValueError(f"Number of embeddings {n_emb} is not divisible by {fig_per_row}")
        for it, (feat_t5, feat_pls) in enumerate(feats):

            name_t5 = t5_feature_names(feat_t5)
            name_pls = pls_feature_names(feat_pls)
            print(f"Plotting feature {feat_t5} vs embeddings ...")
            # if it > 3:
            #     break

            # transforming features to make them comparable
            def transform_t5(x, feat):
                if feat == 0:
                    return x[:, feat] * 2.5
                if feat in [21, 22, 23]:
                    return x[:, feat] # ** -1.0
                return x[:, feat]

            def transform_pls(x, feat):
                if feat == 0:
                    return x[:, feat] * 4.0
                if feat in [7, 8, 9]:
                    return (10 ** x[:, feat]) ** -1.0
                return x[:, feat]

            # transform them (similar)
            f_t5 = transform_t5(sim_x_t5, feat_t5)
            f_pls = transform_pls(sim_x_pls, feat_pls)
            sim_dx = f_t5 - f_pls

            # duplicates (similar)
            fig, axs = plt.subplots(figsize=(16, 8), ncols=fig_per_row, nrows=n_emb // fig_per_row)
            for emb in range(n_emb):
                ax = axs[emb // fig_per_row, emb % fig_per_row]
                _, _, _, im = ax.hist2d(sim_dx, sim_de[:, emb], bins=100, cmap="Blues", cmin=0.5, norm=colors.LogNorm())
                ax.set_xlabel(f"{name_t5} - {name_pls}")
                ax.set_ylabel(f"Embedding {emb} difference")
                ax.set_title(f"{name_t5} - {name_pls} vs Emb. {emb} (duplicates)")
                ax.tick_params(top=True, right=True, which="both", direction="in")
                ax.grid(alpha=0.3)
                fig.colorbar(im, ax=ax, label="Track pairs")
                fig.subplots_adjust(hspace=0.3, wspace=0.3, right=0.98, left=0.05, bottom=0.05, top=0.95)
            pdf.savefig()
            plt.close()

            # transform them (dissimilar)
            f_t5 = transform_t5(dis_x_t5, feat_t5)
            f_pls = transform_pls(dis_x_pls, feat_pls)
            dis_dx = f_t5 - f_pls

            # duplicates (dissimilar)
            fig, axs = plt.subplots(figsize=(16, 8), ncols=fig_per_row, nrows=n_emb // fig_per_row)
            for emb in range(n_emb):
                ax = axs[emb // fig_per_row, emb % fig_per_row]
                _, _, _, im = ax.hist2d(dis_dx, dis_de[:, emb], bins=100, cmap="Reds", cmin=0.5, norm=colors.LogNorm())
                ax.set_xlabel(f"{name_t5} - {name_pls}")
                ax.set_ylabel(f"Embedding {emb} difference")
                ax.set_title(f"{name_t5} - {name_pls} vs Emb. {emb} (non-duplicates)")
                ax.tick_params(top=True, right=True, which="both", direction="in")
                ax.grid(alpha=0.3)
                fig.colorbar(im, ax=ax, label="Track pairs")
                fig.subplots_adjust(hspace=0.3, wspace=0.3, right=0.98, left=0.05, bottom=0.05, top=0.95)
            pdf.savefig()
            plt.close()


def load_root_file_ak(file_path, branches):
    # with uproot.open(f"{file_path}") as fi:
    #     print(fi)
    #     for key in fi.keys():
    #         print(f"File key: {key}")
    with uproot.open(f"{file_path}:tree") as tr:
        print(tr)
        # for key in tr.keys():
        #     print(f"Key: {key}")
        return tr.arrays(branches)


def load_tracking_ntuple(file_path, branches):
    parquet_file = Path("tracking_ntuple.parquet")
    if parquet_file.exists():
        print(f"Loading tracking ntuple from {parquet_file} ...")
        return ak.from_parquet(parquet_file)
    else:
        print(f"Loading tracking ntuple from {file_path} ...")
        with uproot.open(f"{file_path}:trackingNtuple/tree") as tr:
            print(tr)
            data = tr.arrays(branches)
            print(f"Writing tracking ntuple to {parquet_file} ...")
            ak.to_parquet(data, parquet_file)
            return data


def load_t5_t5_pairs():
    with open(PAIRS_T5T5, "rb") as fi:
        data = pickle.load(fi)
    return [
        data["X_left_train"],
        data["X_left_test"],
        data["X_right_train"],
        data["X_right_test"],
        data["y_t5_train"],
        data["y_t5_test"],
        data["w_t5_train"],
        data["w_t5_test"],
        data["true_L_train"],
        data["true_L_test"],
        data["true_R_train"],
        data["true_R_test"]
    ]


def load_t5_pls_pairs():
    with open(PAIRS_T5PLS, "rb") as fi:
        data = pickle.load(fi)
    return [
        data["X_pls_train"],
        data["X_pls_test"],
        data["X_t5raw_train"],
        data["X_t5raw_test"],
        data["y_pls_train"],
        data["y_pls_test"],
        data["w_pls_train"],
        data["w_pls_test"],
    ]


def t5_feature_names(index: int) -> str:
    return [
        "eta1 / 2.5",
        "np.cos(phi1)",
        "np.sin(phi1)",
        "z1 / z_max",
        "r1 / r_max",

        "eta2 - abs(eta1)",
        "delta_phi(phi2, phi1)",
        "(z2 - z1) / z_max",
        "(r2 - r1) / r_max",

        "eta3 - eta2",
        "delta_phi(phi3, phi2)",
        "(z3 - z2) / z_max",
        "(r3 - r2) / r_max",

        "eta4 - eta3",
        "delta_phi(phi4, phi3)",
        "(z4 - z3) / z_max",
        "(r4 - r3) / r_max",

        "eta5 - eta4",
        "delta_phi(phi5, phi4)",
        "(z5 - z4) / z_max",
        "(r5 - r4) / r_max",

        "1.0 / inR",
        "1.0 / brR",
        "1.0 / outR",

        "s1_fake", "s1_prompt", "s1_disp",
        "d_fake",  "d_prompt",  "d_disp",

        # bonus features
        # "ev",
        # "i",
    ][index]


def pls_feature_names(index: int) -> str:
    return [
        "eta/4.0",
        "etaErr/.00139",
        "np.cos(phi)",
        "np.sin(phi)",
        "1.0 / ptIn",
        "np.log10(ptErr)",
        "isQuad",
        "np.log10(circleCenterX)",
        "np.log10(circleCenterY)",
        "np.log10(circleRadius)",
    ][index]


if __name__ == "__main__":
    main()
