import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
from matplotlib import rcParams
rcParams["font.size"] = 16

from preprocess import load_t5_features, load_pls_features
from preprocess import load_t5_t5_pairs, load_t5_pls_pairs
from ml import EmbeddingNetT5, EmbeddingNetpLS

BONUS_FEATURES = 2

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
    return parser.parse_args()


def main():
    args = options()
    pdf_name = args.pdf
    features_t5 = args.features_t5
    features_pls = args.features_pls
    pairs_t5t5 = args.pairs_t5t5
    pairs_t5pls = args.pairs_t5pls
    model_weights = args.model
    other_model = args.other_model

    print(f"Loading T5-T5 pairs from {pairs_t5t5}")
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
    true_R_test) = load_t5_t5_pairs(pairs_t5t5)
    print(X_left_test.shape)

    print(f"Loading T5-PLS pairs from {pairs_t5pls}")
    (X_pls_train,
     X_pls_test,
     X_t5raw_train,
     X_t5raw_test,
     y_pls_train,
     y_pls_test,
     w_pls_train,
     w_pls_test) = load_t5_pls_pairs(pairs_t5pls)


    print("Loading embedding networks")
    embed_t5 = EmbeddingNetT5()
    embed_pls = EmbeddingNetpLS()
    if not model_weights:
        print("No model weights provided, using header weights")
        embed_t5.load_from_header()
        embed_pls.load_from_header()
    else:
        print(f"Loading model weights from {model_weights}")
        checkpoint = torch.load(model_weights)
        embed_t5.load_state_dict(checkpoint["embed_t5"])
        embed_pls.load_state_dict(checkpoint["embed_pls"])
    embed_t5.eval()
    embed_pls.eval()

    if other_model:
        print(f"Loading other model weights from {other_model}")
        other_checkpoint = torch.load(other_model)
        embed_t5_other = EmbeddingNetT5()
        embed_pls_other = EmbeddingNetpLS()
        embed_t5_other.load_state_dict(other_checkpoint["embed_t5"])
        embed_pls_other.load_state_dict(other_checkpoint["embed_pls"])
        embed_t5_other.eval()
        embed_pls_other.eval()

    print(f"Choosing dataset for PCA: {args.pca_dataset}")
    if args.pca_dataset == "train":
        x_t5 = X_left_train
        x_pls = X_pls_train
    elif args.pca_dataset == "test":
        x_t5 = X_left_test
        x_pls = X_pls_test
    else:
        raise ValueError("Invalid dataset choice for PCA. Choose 'train' or 'test'.")

    print("Embedding T5s and PLSs")
    x_t5 = torch.tensor(x_t5[:, :-BONUS_FEATURES], requires_grad=False)
    x_pls = torch.tensor(x_pls[:, :-BONUS_FEATURES], requires_grad=False)
    embedded_t5 = embed_t5(x_t5).detach().numpy()
    embedded_pls = embed_pls(x_pls).detach().numpy()
    print(f"Embedded T5s shape: {embedded_t5.shape}")
    print(f"Embedded PLSs shape: {embedded_pls.shape}")
    if other_model:
        print("Embedding T5s and PLSs with other model")
        embedded_t5_other = embed_t5_other(x_t5).detach().numpy()
        embedded_pls_other = embed_pls_other(x_pls).detach().numpy()
        print(f"Other Embedded T5s shape: {embedded_t5_other.shape}")
        print(f"Other Embedded PLSs shape: {embedded_pls_other.shape}")

    # add engineered features to the feature vectors
    x_t5 = x_t5.detach().numpy()
    x_pls = x_pls.detach().numpy()
    if args.engineer:
        eng_t5_eta = x_t5[:, 0] * 2.5
        eng_t5_phi = np.arctan2(x_t5[:, 2], x_t5[:, 1])
        eng_pls_eta = x_pls[:, 0] * 4.0
        eng_pls_phi = np.arctan2(x_pls[:, 3], x_pls[:, 2])
        x_t5 = np.concatenate((x_t5,
                               eng_t5_eta.reshape(-1, 1),
                               eng_t5_phi.reshape(-1, 1),
                               ), axis=1)
        x_pls = np.concatenate((x_pls,
                                eng_pls_eta.reshape(-1, 1),
                                eng_pls_phi.reshape(-1, 1),
                                ), axis=1)


    # bookkeeping
    t5s = slice(0, len(embedded_t5))
    pls = slice(len(embedded_t5), len(embedded_t5) + len(embedded_pls))

    # do PCA
    print("Performing PCA on embedded T5s and PLSs")
    input = np.concatenate((embedded_t5, embedded_pls))
    pca = PCA(n_components=args.n_pca)
    proj = pca.fit_transform(input)
    print(f"Combined PCA projection shape: {proj.shape}")
    if other_model:
        print("Performing PCA on other model's embedded T5s and PLSs")
        input_other = np.concatenate((embedded_t5_other, embedded_pls_other))
        pca_other = PCA(n_components=args.n_pca)
        proj_other = pca_other.fit_transform(input_other)
        print(f"Other model PCA projection shape: {proj_other.shape}")

    # PCA results
    print("PCA results:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Explained variance: {pca.explained_variance_}")
    print(f"Principal components shape: {pca.components_.shape}")

    # Check by hand
    if args.checkmath:
        ncheck = 2
        print(f"Principal components:")
        print(f"{pca.components_}")
        print(f"Mean of the input data: {pca.mean_}")
        print(f"Embedded T5s shape and first {ncheck}: {embedded_t5.shape}")
        print(embedded_t5[:ncheck])
        print(f"Embedded PLSs shape and first {ncheck}: {embedded_pls.shape}")
        print(embedded_pls[:ncheck])
        print(f"PCA projection shape and first {ncheck} T5s: {proj[t5s].shape}")
        print(proj[t5s][:ncheck])
        print(f"PCA projection shape and first {ncheck} PLSs: {proj[pls].shape}")
        print(proj[pls][:ncheck])

    # t-SNE? On the todo list
    if args.tsne:
        print("Performing t-SNE")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_t5 = tsne.fit_transform(embedded_t5)
        tsne_pls = tsne.fit_transform(embedded_pls)

    # plot options
    cmap = "hot"
    cmin = 0.5
    pad = 0.01

    print("Plotting!")
    with PdfPages(pdf_name) as pdf:

        # feature correlation check
        for dim in range(args.n_pca):
            print(f"Plotting PCA Component {dim} correlations with features")
            for (name, slc, sample) in [
                ("T5", t5s, x_t5),
                ("PLS", pls, x_pls),
            ]:
                if args.quickplot and dim > 0:
                    break
                n_features = sample.shape[1]
                for feature in range(n_features):
                    #if args.quickplot and feature > 5:
                    #    break
                    feat_name = feature_name(name, feature)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    _, _, _, im = ax.hist2d(proj[slc][:, dim], sample[:, feature], bins=100, cmap=cmap, cmin=cmin)
                    ax.set_xlabel(f"PCA Component {dim}")
                    ax.set_ylabel(f"{name} Feature {feature}: {feat_name}")
                    ax.set_title(f"PCA Component {dim} vs {name} Feature {feature}")
                    ax.tick_params(right=True, top=True, which="both", direction="in")
                    ax.text(1.08, 1.02, "Tracks", transform=ax.transAxes)
                    fig.colorbar(im, ax=ax, pad=pad)
                    fig.subplots_adjust(right=0.98, left=0.13, bottom=0.09, top=0.95)
                    pdf.savefig()
                    plt.close()


        # other model?
        if other_model:

            # everything at once
            print(f"Plotting PCA Component i vs j for model vs other model")
            fig, ax = plt.subplots(figsize=(40, 40), nrows=args.n_pca, ncols=args.n_pca)
            for dim_i in range(args.n_pca):
                for dim_j in range(dim_i, args.n_pca):
                    corr = np.corrcoef(proj[:, dim_i], proj_other[:, dim_j])[0, 1]
                    _, _, _, im = ax[dim_i, dim_j].hist2d(proj[:, dim_i], proj_other[:, dim_j], bins=100, cmap=cmap, cmin=cmin)
                    ax[dim_i, dim_j].set_xlabel(f"Model PCA Component {dim_i}")
                    ax[dim_i, dim_j].set_ylabel(f"Other Model PCA Component {dim_j}")
                    ax[dim_i, dim_j].tick_params(right=True, top=True, which="both", direction="in")
                    ax[dim_i, dim_j].text(0.50, 1.02, f"Corr: {corr:.2f}", transform=ax[dim_i, dim_j].transAxes, ha="center")
            fig.subplots_adjust(right=0.98, left=0.03, bottom=0.03, top=0.97, wspace=0.3, hspace=0.3)
            pdf.savefig()
            plt.close()

            # only diagonal terms
            for dim_i in range(args.n_pca):
                print(f"Plotting PCA Component {dim_i} vs {dim_i} for model vs other model")
                fig, ax = plt.subplots(figsize=(8, 8))
                _, _, _, im = ax.hist2d(proj[:, dim_i], proj_other[:, dim_i], bins=100, cmap=cmap, cmin=cmin)
                ax.set_xlabel(f"Model PCA Component {dim_i}")
                ax.set_ylabel(f"Other Model PCA Component {dim_i}")
                ax.set_title(f"Model vs Other Model PCA Component {dim_i}")
                ax.tick_params(right=True, top=True, which="both", direction="in")
                ax.text(1.08, 1.02, "Tracks", transform=ax.transAxes)
                fig.colorbar(im, ax=ax, pad=pad)
                fig.subplots_adjust(right=0.98, left=0.13, bottom=0.09, top=0.95)
                pdf.savefig()
                plt.close()


        # sample checks
        for (proj_data, title) in [(proj[t5s], "PCA Projection: T5s"),
                                   (proj[pls], "PCA Projection: PLSs"),
                                   (proj, "PCA Projection: T5s and PLSs"),
                                   ]:
            print(f"Plotting {title}")
            for norm in [None, colors.LogNorm()]:
                fig, ax = plt.subplots(figsize=(8, 8))
                _, _, _, im = ax.hist2d(proj_data[:, 0], proj_data[:, 1], bins=100, cmap=cmap, cmin=cmin, norm=norm)
                ax.set_xlabel("PCA Component 0")
                ax.set_ylabel("PCA Component 1")
                ax.set_title(title)
                ax.text(1.08, 1.02, "Tracks", transform=ax.transAxes)
                ax.tick_params(right=True, top=True, which="both", direction="in")
                fig.colorbar(im, ax=ax, pad=pad)
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

    else:
        raise ValueError(f"Unknown PLS feature index: {feature}")



if __name__ == "__main__":
    main()
