import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
from matplotlib import rcParams
rcParams["font.size"] = 16

from preprocess import load_t5_features, load_pls_features
from preprocess import load_t5_t5_pairs, load_t5_pls_pairs
from ml import EmbeddingNetT5, EmbeddingNetpLS

BONUS_FEATURES = 2
k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="",
                        help="Path to save or load the model weights")
    parser.add_argument("--other_model", type=str, default="",
                        help="Path to save or load the model weights for comparison")
    parser.add_argument("--pdf", type=str, default="pca.pdf",
                        help="Path to save the output plots in PDF format")
    parser.add_argument("--features_t5", type=str, default="features_t5.pkl",
                        help="Path to the precomputed T5 features file")
    parser.add_argument("--features_pls", type=str, default="features_pls.pkl",
                        help="Path to the precomputed PLS features file")
    parser.add_argument("--pairs_t5t5", type=str, default="pairs_t5t5.pkl",
                        help="Path to the precomputed T5-T5 pairs file")
    parser.add_argument("--pairs_t5pls", type=str, default="pairs_t5pls.pkl",
                        help="Path to the precomputed T5-PLS pairs file")
    parser.add_argument("--pca_dataset", type=str, default="test",
                        help="Choice of dataset for PCA: 'train' or 'test'")
    parser.add_argument("--n_pca", type=int, default=6,
                        help="Number of PCA components")
    parser.add_argument("--tsne", action='store_true',
                        help="Whether to perform t-SNE on the PCA projections")
    parser.add_argument("--quickplot", action='store_true',
                        help="Only make a couple plots for testing purposes")
    parser.add_argument("--checkmath", action='store_true',
                        help="Check the PCA math by printing some intermediate results")
    parser.add_argument("--engineer", action='store_true',
                        help="Engineer additional features for T5s and PLSs")
    parser.add_argument("--draw_envelope", action='store_true',
                        help="Draw envelope around 2d difference plots")
    return parser.parse_args()


def main():
    args = options()
    plotter = PCAPlotter(pdf_name=args.pdf,
                         features_t5=args.features_t5,
                         features_pls=args.features_pls,
                         pairs_t5t5=args.pairs_t5t5,
                         pairs_t5pls=args.pairs_t5pls,
                         model_weights=args.model,
                         other_model=args.other_model,
                         n_pca=args.n_pca,
                         pca_dataset=args.pca_dataset,
                         engineer=args.engineer,
                         checkmath=args.checkmath,
                         tsne=args.tsne,
                         quickplot=args.quickplot,
                         draw_envelope=args.draw_envelope,
                         )
    plotter.load_data()
    plotter.load_embedding_networks()
    plotter.embed_data()
    plotter.engineer_features()
    plotter.do_pca()
    plotter.check_math()
    plotter.do_tsne()
    plotter.plot()


class PCAPlotter:

    def __init__(self,
                 pdf_name,
                 features_t5,
                 features_pls,
                 pairs_t5t5,
                 pairs_t5pls,
                 model_weights,
                 other_model,
                 n_pca,
                 pca_dataset,
                 engineer,
                 checkmath,
                 tsne,
                 quickplot,
                 draw_envelope,
                 ):

        self.pdf_name = pdf_name
        self.features_t5 = features_t5
        self.features_pls = features_pls
        self.pairs_t5t5 = pairs_t5t5
        self.pairs_t5pls = pairs_t5pls
        self.model_weights = model_weights
        self.other_model = other_model
        self.n_pca = n_pca
        self.pca_dataset = pca_dataset
        self.engineer = engineer
        self.checkmath = checkmath
        self.tsne = tsne
        self.quickplot = quickplot
        self.draw_envelope = draw_envelope


    def load_data(self):

        print(f"Loading T5-T5 pairs from {self.pairs_t5t5}")
        (X_left_train,
         X_left_test,
         X_right_train,
         X_right_test,
         y_t5_train,
         y_t5_test,
         w_t5_train,
         w_t5_test,
         true_L_train,
         true_L_test,
         true_R_train,
         true_R_test) = load_t5_t5_pairs(self.pairs_t5t5)
        # print(X_left_test.shape)

        print(f"Loading T5-PLS pairs from {self.pairs_t5pls}")
        (X_pls_train,
         X_pls_test,
         X_t5raw_train,
         X_t5raw_test,
         y_pls_train,
         y_pls_test,
         w_pls_train,
         w_pls_test) = load_t5_pls_pairs(self.pairs_t5pls)

        print(f"Choosing dataset for PCA: {self.pca_dataset}")
        if self.pca_dataset == "train":
            self.x_t5 = X_left_train
            self.x_pls = X_pls_train
        elif self.pca_dataset == "test":
            self.x_t5 = X_left_test
            self.x_pls = X_pls_test
        else:
            raise ValueError("Invalid dataset choice for PCA. Choose 'train' or 'test'.")

        # Save pairwise data, too
        self.x_left_test = X_left_test
        self.x_right_test = X_right_test
        self.y_t5_test = y_t5_test
        self.x_pls_test = X_pls_test
        self.x_t5_test = X_t5raw_test
        self.y_pls_test = y_pls_test
        self.x_diff_test = (self.x_left_test - self.x_right_test)[:, :-BONUS_FEATURES]


    def load_embedding_networks(self):

        print("Loading embedding networks")
        self.embed_t5 = EmbeddingNetT5()
        self.embed_pls = EmbeddingNetpLS()
        if not self.model_weights:
            print("No model weights provided, using header weights")
            self.embed_t5.load_from_header()
            self.embed_pls.load_from_header()
        else:
            print(f"Loading model weights from {self.model_weights}")
            checkpoint = torch.load(self.model_weights)
            self.embed_t5.load_state_dict(checkpoint["embed_t5"])
            self.embed_pls.load_state_dict(checkpoint["embed_pls"])
        self.embed_t5.eval()
        self.embed_pls.eval()

        if self.other_model:
            print(f"Loading other model weights from {self.other_model}")
            other_checkpoint = torch.load(self.other_model)
            self.embed_t5_other = EmbeddingNetT5()
            self.embed_pls_other = EmbeddingNetpLS()
            self.embed_t5_other.load_state_dict(other_checkpoint["embed_t5"])
            self.embed_pls_other.load_state_dict(other_checkpoint["embed_pls"])
            self.embed_t5_other.eval()
            self.embed_pls_other.eval()


    def embed_data(self):

        def embed(x, net):
            x_tensor = torch.tensor(x[:, :-BONUS_FEATURES], requires_grad=False)
            return net(x_tensor).detach().numpy()

        print("Embedding T5s and PLSs")
        self.embedded_t5 = embed(self.x_t5, self.embed_t5)
        self.embedded_pls = embed(self.x_pls, self.embed_pls)
        print(f"Embedded T5s shape: {self.embedded_t5.shape}")
        print(f"Embedded PLSs shape: {self.embedded_pls.shape}")
        if self.other_model:
            print("Embedding T5s and PLSs with other model")
            self.embedded_t5_other = embed(self.x_t5, self.embed_t5_other)
            self.embedded_pls_other = embed(self.x_pls, self.embed_pls_other)
            print(f"Other Embedded T5s shape: {self.embedded_t5_other.shape}")
            print(f"Other Embedded PLSs shape: {self.embedded_pls_other.shape}")

        # bookkeeping
        self.t5s = slice(0, len(self.embedded_t5))
        self.pls = slice(len(self.embedded_t5), len(self.embedded_t5) + len(self.embedded_pls))

        # Save pairwise data, too
        self.emb_x_l = embed(self.x_left_test, self.embed_t5)
        self.emb_x_r = embed(self.x_right_test, self.embed_t5)
        self.emb_x_pls = embed(self.x_pls_test, self.embed_pls)
        self.emb_x_t5 = embed(self.x_t5_test, self.embed_t5)
        self.d_t5t5 = np.linalg.norm(self.emb_x_l - self.emb_x_r, axis=1)
        self.d_t5pls = np.linalg.norm(self.emb_x_t5 - self.emb_x_pls, axis=1)


    def engineer_features(self):

        # add engineered features to the feature vectors
        if self.engineer:

            # T5
            eng_t5_eta = self.x_t5[:, 0] * 2.5
            eng_t5_phi = np.arctan2(self.x_t5[:, 2], self.x_t5[:, 1])
            eng_t5_pt = (self.x_t5[:, 21] ** -1) * k2Rinv1GeVf * 2
            eng_t5_max_disp = np.maximum(self.x_t5[:, 26], self.x_t5[:, 26] + self.x_t5[:, 29])

            # PLS
            eng_pls_eta = self.x_pls[:, 0] * 4.0
            eng_pls_phi = np.arctan2(self.x_pls[:, 3], self.x_pls[:, 2])
            eng_pls_rinv = (10 ** self.x_pls[:, 9]) ** -1.0
            eng_pls_pt = self.x_pls[:, 4] ** -1
            eng_pls_center_r = np.sqrt((10 ** self.x_pls[:, 7]) ** 2 + (10 ** self.x_pls[:, 8]) ** 2)

            # Combine
            self.x_t5 = np.concatenate((self.x_t5,
                                        eng_t5_eta.reshape(-1, 1),
                                        eng_t5_phi.reshape(-1, 1),
                                        eng_t5_pt.reshape(-1, 1),
                                        eng_t5_max_disp.reshape(-1, 1),
                                        ), axis=1)
            self.x_pls = np.concatenate((self.x_pls,
                                         eng_pls_eta.reshape(-1, 1),
                                         eng_pls_phi.reshape(-1, 1),
                                         eng_pls_rinv.reshape(-1, 1),
                                         eng_pls_pt.reshape(-1, 1),
                                         eng_pls_center_r.reshape(-1, 1),
                                         ), axis=1)


    def do_pca(self):

        # do PCA
        print("Performing PCA on embedded T5s and PLSs")
        input = np.concatenate((self.embedded_t5, self.embedded_pls))
        self.pca = PCA(n_components=self.n_pca)
        self.proj = self.pca.fit_transform(input)
        print(f"Combined PCA projection shape: {self.proj.shape}")
        if self.other_model:
            print("Performing PCA on other model's embedded T5s and PLSs")
            input_other = np.concatenate((self.embedded_t5_other, self.embedded_pls_other))
            self.pca_other = PCA(n_components=self.n_pca)
            self.proj_other = self.pca_other.fit_transform(input_other)
            print(f"Other model PCA projection shape: {self.proj_other.shape}")

        # PCA results
        print("PCA results:")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Explained variance: {self.pca.explained_variance_}")
        print(f"Principal components shape: {self.pca.components_.shape}")

        # More PCA
        self.proj_x_l = self.pca.transform(self.emb_x_l)
        self.proj_x_r = self.pca.transform(self.emb_x_r)


    def check_math(self):

        # Check by hand
        if self.checkmath:
            ncheck = 2
            print(f"Principal components:")
            print(f"{self.pca.components_}")
            print(f"Mean of the input data: {self.pca.mean_}")
            print(f"Embedded T5s shape and first {ncheck}: {self.embedded_t5.shape}")
            print(self.embedded_t5[:ncheck])
            print(f"Embedded PLSs shape and first {ncheck}: {self.embedded_pls.shape}")
            print(self.embedded_pls[:ncheck])
            print(f"PCA projection shape and first {ncheck} T5s: {self.proj[self.t5s].shape}")
            print(self.proj[self.t5s][:ncheck])
            print(f"PCA projection shape and first {ncheck} PLSs: {self.proj[self.pls].shape}")
            print(self.proj[self.pls][:ncheck])

        # Check if PCA preserves the distances
        if self.checkmath:
            x1 = self.embedded_t5[:5]
            x2 = self.embedded_t5[5:10]
            distance_emb = np.linalg.norm(x1 - x2, axis=1)
            distance_pca = np.linalg.norm(self.pca.transform(x1) - self.pca.transform(x2), axis=1)
            print("Distance check:")
            print(f"Distances in embedded space: {distance_emb}")
            print(f"Distances in PCA space: {distance_pca}")


    def do_tsne(self):

        # t-SNE? On the todo list
        if self.tsne:
            print("Performing t-SNE")
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            tsne_t5 = tsne.fit_transform(self.embedded_t5)
            tsne_pls = tsne.fit_transform(self.embedded_pls)


    def plot(self):

        # plot options
        self.cmap = "hot"
        self.cmin = 0.5
        self.pad = 0.01
        bins = 100

        print("Plotting!")
        with PdfPages(self.pdf_name) as pdf:

            dup, nodup = 0, 1
            dims = range(-1, 6) # self.n_pca
            bins = {}
            bins["d"] = np.logspace(-3, 1, 101)

            for status in [dup, nodup]:

                print(f"Plotting T5-T5 pairs for status {status}")
                mask = self.y_t5_test == status
                title = "duplicate" if status == dup else "non-duplicate"

                for dim in dims:

                    if self.quickplot and dim > 1:
                        break

                    print(f"Plotting PCA dim {dim} for T5-T5 pairs")

                    # the distance to plot
                    if dim == -1:
                        dist = self.d_t5t5
                    else:
                        dist = self.proj_x_l[:, dim] - self.proj_x_r[:, dim]

                    for feature in range(self.x_diff_test.shape[1]):
                    # for feature in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    # for feature in [0]:
                        if self.quickplot and feature > 2:
                            break
                        fig, ax = plt.subplots(figsize=(8, 8))
                        feat_mask = mask & (self.x_diff_test[:, feature] != 0)
                        feat_name = feature_name("T5", feature)
                        numer, denom = np.sum(mask & (self.x_diff_test[:, feature] == 0)), np.sum(mask)
                        excluded = f"{int(100*numer/denom)}%"
                        counts, xbins, ybins, im = ax.hist2d(dist[feat_mask],
                                                             self.x_diff_test[feat_mask, feature],
                                                             bins=(bins["d"], 100),
                                                             cmap=self.cmap, cmin=self.cmin)

                        if self.draw_envelope:
                            x_bin_centers, percentile_lo, percentile_hi = get_bounds_of_thing(dist[feat_mask],
                                                                                              self.x_diff_test[feat_mask, feature],
                                                                                              bins["d"])
                            ax.scatter(x_bin_centers, percentile_lo, marker="_", color='cyan', linewidth=1.2, label="25th-75th percentile")
                            ax.scatter(x_bin_centers, percentile_hi, marker="_", color='cyan', linewidth=1.2)

                        xlabel = "total" if dim == -1 else f"PCA dim. {dim}"
                        ax.set_xlabel(f"Euclidean distance, {xlabel}")
                        ax.set_ylabel(f"Left - Right, feature {feature}: {feat_name}")
                        ax.set_title(f"T5-T5 pairs: {title}. Excluding {excluded}")
                        ax.tick_params(right=True, top=True, which="both", direction="in")
                        ax.text(1.08, 1.02, "Pairs", transform=ax.transAxes)
                        ax.semilogx()
                        fig.colorbar(im, ax=ax, pad=self.pad)
                        fig.subplots_adjust(right=0.98, left=0.16, bottom=0.09, top=0.95)
                        pdf.savefig()
                        plt.close()

            return

            # PCA components
            print("Plotting PCA components")
            fig, ax = plt.subplots(figsize=(8, 8))
            for dim in range(self.n_pca):
                ax.hist(self.proj[:, dim], bins=bins, label=f"PCA Component {dim}",
                        histtype="step", lw=2, color=f"C{dim}")
            ax.set_xlabel("Value in PCA embedding")
            ax.set_ylabel("Tracks")
            ax.set_title("PCA Components")
            ax.legend()
            ax.tick_params(right=True, top=True, which="both", direction="in")
            fig.subplots_adjust(right=0.98, left=0.16, bottom=0.09, top=0.95)
            pdf.savefig()
            plt.close()

            # Correlations of different PCA components
            print(f"Plotting PCA Component i vs j")
            fig, ax = plt.subplots(figsize=(40, 40), nrows=self.n_pca, ncols=self.n_pca)
            for dim_i in range(self.n_pca):
                for dim_j in range(dim_i, self.n_pca):
                    corr = np.corrcoef(self.proj[:, dim_i], self.proj[:, dim_j])[0, 1]
                    _, _, _, im = ax[dim_i, dim_j].hist2d(self.proj[:, dim_i], self.proj[:, dim_j], bins=bins, cmap=self.cmap, cmin=self.cmin)
                    ax[dim_i, dim_j].set_xlabel(f"PCA Component {dim_i}")
                    ax[dim_i, dim_j].set_ylabel(f"PCA Component {dim_j}")
                    ax[dim_i, dim_j].tick_params(right=True, top=True, which="both", direction="in")
                    ax[dim_i, dim_j].text(0.50, 1.02, f"Corr: {corr:.2f}", transform=ax[dim_i, dim_j].transAxes, ha="center")
            fig.subplots_adjust(right=0.98, left=0.03, bottom=0.03, top=0.97, wspace=0.3, hspace=0.3)
            pdf.savefig()
            plt.close()

            # feature correlation check
            for dim in range(self.n_pca):
                print(f"Plotting PCA Component {dim} correlations with features")
                for (name, slc, sample) in [
                    ("T5", self.t5s, self.x_t5),
                    ("PLS", self.pls, self.x_pls),
                ]:
                    if self.quickplot and dim > 2:
                        break
                    n_features = sample.shape[1]
                    for feature in range(n_features):
                        if self.quickplot and feature > 5:
                            break
                        feat_name = feature_name(name, feature)
                        this_bins = feature_binning(dim, feat_name)
                        fig, ax = plt.subplots(figsize=(8, 8))
                        _, _, _, im = ax.hist2d(self.proj[slc][:, dim], sample[:, feature], bins=this_bins, cmap=self.cmap, cmin=self.cmin)
                        ax.set_xlabel(f"PCA Component {dim}")
                        ax.set_ylabel(f"{name} Feature {feature}: {feat_name}")
                        ax.set_title(f"PCA Component {dim} vs {name} Feature {feature}")
                        ax.tick_params(right=True, top=True, which="both", direction="in")
                        ax.text(1.08, 1.02, "Tracks", transform=ax.transAxes)
                        fig.colorbar(im, ax=ax, pad=self.pad)
                        fig.subplots_adjust(right=0.98, left=0.13, bottom=0.09, top=0.95)
                        pdf.savefig()
                        plt.close()


            # other model?
            if self.other_model:

                # everything at once
                print(f"Plotting PCA Component i vs j for model vs other model")
                fig, ax = plt.subplots(figsize=(40, 40), nrows=self.n_pca, ncols=self.n_pca)
                for dim_i in range(self.n_pca):
                    for dim_j in range(dim_i, self.n_pca):
                        corr = np.corrcoef(self.proj[:, dim_i], self.proj_other[:, dim_j])[0, 1]
                        _, _, _, im = ax[dim_i, dim_j].hist2d(self.proj[:, dim_i], self.proj_other[:, dim_j], bins=bins, cmap=self.cmap, cmin=self.cmin)
                        ax[dim_i, dim_j].set_xlabel(f"Model PCA Component {dim_i}")
                        ax[dim_i, dim_j].set_ylabel(f"Other Model PCA Component {dim_j}")
                        ax[dim_i, dim_j].tick_params(right=True, top=True, which="both", direction="in")
                        ax[dim_i, dim_j].text(0.50, 1.02, f"Corr: {corr:.2f}", transform=ax[dim_i, dim_j].transAxes, ha="center")
                fig.subplots_adjust(right=0.98, left=0.03, bottom=0.03, top=0.97, wspace=0.3, hspace=0.3)
                pdf.savefig()
                plt.close()

                # only diagonal terms
                for dim_i in range(self.n_pca):
                    print(f"Plotting PCA Component {dim_i} vs {dim_i} for model vs other model")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    _, _, _, im = ax.hist2d(self.proj[:, dim_i], self.proj_other[:, dim_i], bins=bins, cmap=self.cmap, cmin=self.cmin)
                    ax.set_xlabel(f"Model PCA Component {dim_i}")
                    ax.set_ylabel(f"Other Model PCA Component {dim_i}")
                    ax.set_title(f"Model vs Other Model PCA Component {dim_i}")
                    ax.tick_params(right=True, top=True, which="both", direction="in")
                    ax.text(1.08, 1.02, "Tracks", transform=ax.transAxes)
                    fig.colorbar(im, ax=ax, pad=self.pad)
                    fig.subplots_adjust(right=0.98, left=0.13, bottom=0.09, top=0.95)
                    pdf.savefig()
                    plt.close()


            # sample checks
            for (proj_data, title) in [(self.proj[self.t5s], "PCA Projection: T5s"),
                                       (self.proj[self.pls], "PCA Projection: PLSs"),
                                       (self.proj, "PCA Projection: T5s and PLSs"),
                                       ]:
                print(f"Plotting {title}")
                for norm in [None, colors.LogNorm()]:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    _, _, _, im = ax.hist2d(self.proj_data[:, 0], self.proj_data[:, 1], bins=bins, cmap=self.cmap, cmin=self.cmin, norm=norm)
                    ax.set_xlabel("PCA Component 0")
                    ax.set_ylabel("PCA Component 1")
                    ax.set_title(title)
                    ax.text(1.08, 1.02, "Tracks", transform=ax.transAxes)
                    ax.tick_params(right=True, top=True, which="both", direction="in")
                    fig.colorbar(im, ax=ax, pad=self.pad)
                    fig.subplots_adjust(right=0.98, left=0.12, bottom=0.09, top=0.95)
                    pdf.savefig()
                    plt.close()


    # get t5s
    # get pls
    # embed T5s
    # embed PLSs
    # perform PCA on embedded T5s and PLSs
    # Plot T5s in lower dimensions
    # Plot T5s at low pt, or high eta, or displaced, etc
    # Look for trends
    # Look at PCA decomposition


def feature_binning(dim: int, feat_name: str):
    bins = 100
    if dim == 0 and feat_name == "pT":
        return [bins, np.linspace(0.4, 5, 100)]
    elif dim == 1 and feat_name == "pT":
        return [np.linspace(-3, 2.5, 100), np.linspace(0.4, 5, 100)]
    elif feat_name == "circleCenterR":
        if dim == 1:
            return [np.linspace(-3, 2.5, 100), np.linspace(0, 300, 100)]
        return [bins, np.linspace(0, 300, 100)]
    return bins

def feature_name(name: str, feature: int) -> str:
    if name == "T5":
        return feature_name_t5(feature)
    elif name == "PLS":
        return feature_name_pls(feature)
    else:
        raise ValueError(f"Unknown feature type: {name}")


def feature_name_t5(feature: int) -> str:
    if feature == 0:
        return "eta1 / 2.5"
    elif feature == 1:
        return "np.cos(phi1)"
    elif feature == 2:
        return "np.sin(phi1)"
    elif feature == 3:
        return "z1 / z_max"
    elif feature == 4:
        return "r1 / r_max"

    elif feature == 5:
        return "eta2 - abs(eta1)"
    elif feature == 6:
        return "delta_phi(phi2, phi1)"
    elif feature == 7:
        return "(z2 - z1) / z_max"
    elif feature == 8:
        return "(r2 - r1) / r_max"

    elif feature == 9:
        return "eta3 - eta2"
    elif feature == 10:
        return "delta_phi(phi3, phi2)"
    elif feature == 11:
        return "(z3 - z2) / z_max"
    elif feature == 12:
        return "(r3 - r2) / r_max"

    elif feature == 13:
        return "eta4 - eta3"
    elif feature == 14:
        return "delta_phi(phi4, phi3)"
    elif feature == 15:
        return "(z4 - z3) / z_max"
    elif feature == 16:
        return "(r4 - r3) / r_max"

    elif feature == 17:
        return "eta5 - eta4"
    elif feature == 18:
        return "delta_phi(phi5, phi4)"
    elif feature == 19:
        return "(z5 - z4) / z_max"
    elif feature == 20:
        return "(r5 - r4) / r_max"

    elif feature == 21:
        return "1.0 / inR"
    elif feature == 22:
        return "1.0 / brR"
    elif feature == 23:
        return "1.0 / outR"

    elif feature == 24:
        return "s1_fake"
    elif feature == 25:
        return "s1_prompt"
    elif feature == 26:
        return "s1_disp"
    elif feature == 27:
        return "d_fake"
    elif feature == 28:
        return "d_prompt"
    elif feature == 29:
        return "d_disp"

    # Feature engineering begins here
    elif feature == 30:
        return "eta"
    elif feature == 31:
        return "phi"
    elif feature == 32:
        return "pT"
    elif feature == 33:
        return "max(disp_Inner, disp_Outer)"

    else:
        raise ValueError(f"Unknown T5 feature index: {feature}")


def feature_name_pls(feature: int) -> str:
    if feature == 0:
        return "eta/4.0"
    elif feature == 1:
        return "etaErr/.00139"
    elif feature == 2:
        return "np.cos(phi)"
    elif feature == 3:
        return "np.sin(phi)"
    elif feature == 4:
        return "1.0 / ptIn"
    elif feature == 5:
        return "np.log10(ptErr)"
    elif feature == 6:
        return "isQuad"
    elif feature == 7:
        return "np.log10(circleCenterX)"
    elif feature == 8:
        return "np.log10(circleCenterY)"
    elif feature == 9:
        return "np.log10(circleRadius)"

    # Feature engineering begins here
    elif feature == 10:
        return "eta"
    elif feature == 11:
        return "phi"
    elif feature == 12:
        return "1 / R"
    elif feature == 13:
        return "pT"
    elif feature == 14:
        return "circleCenterR"

    else:
        raise ValueError(f"Unknown PLS feature index: {feature}")

def get_bounds_of_thing(x, y, xbins):
    percentile_25 = []
    percentile_75 = []
    x_bin_centers = []

    for i in range(len(xbins) - 1):
        x_mask = (x >= xbins[i]) & (x < xbins[i+1])
        y_in_bin = y[x_mask]

        if len(y_in_bin) >= 100:  # Avoid percentile calc on 0/1 point
            p25, p75 = np.percentile(y_in_bin, [25, 75])
            percentile_25.append(p25)
            percentile_75.append(p75)
            x_bin_centers.append(0.5 * (xbins[i] + xbins[i+1]))

    return np.array(x_bin_centers), np.array(percentile_25), np.array(percentile_75)



if __name__ == "__main__":
    main()
