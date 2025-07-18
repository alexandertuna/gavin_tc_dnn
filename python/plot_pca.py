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
    return parser.parse_args()


def main():
    args = options()
    pdf_name = args.pdf
    features_t5 = args.features_t5
    features_pls = args.features_pls
    pairs_t5t5 = args.pairs_t5t5
    pairs_t5pls = args.pairs_t5pls
    model_weights = args.model

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

    # bookkeeping
    t5s = slice(0, len(embedded_t5))
    pls = slice(len(embedded_t5), len(embedded_t5) + len(embedded_pls))

    print("Performing PCA on embedded T5s and PLSs")
    input = np.concatenate((embedded_t5, embedded_pls))
    proj = PCA(n_components=args.n_pca).fit_transform(input)
    # pca_t5 = PCA(n_components=args.n_pca)
    # pca_pls = PCA(n_components=args.n_pca)
    # proj_t5 = pca_t5.fit_transform(embedded_t5)
    # proj_pls = pca_pls.fit_transform(embedded_pls)
    # print(f"PCA T5 projection shape: {proj_t5.shape}")
    # print(f"PCA PLS projection shape: {proj_pls.shape}")
    print(f"Combined PCA projection shape: {proj.shape}")

    if args.tsne:
        print("Performing t-SNE")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_t5 = tsne.fit_transform(embedded_t5)
        tsne_pls = tsne.fit_transform(embedded_pls)

    print("Plotting!")
    with PdfPages(pdf_name) as pdf:

        # feature correlation check
        for dim in range(args.n_pca):
            print(f"Plotting PCA Component {dim} correlations with features")
            for (name, slc, sample) in [
                ("T5", t5s, x_t5),
                ("PLS", pls, x_pls),
            ]:
                n_features = sample.shape[1]
                for feature in range(n_features):
                    # if feature > 5:
                    #     break
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.hist2d(proj[slc][:, dim], sample[:, feature], bins=100, cmap='gist_heat', cmin=0.5)
                    ax.set_xlabel(f"PCA Component {dim}")
                    ax.set_ylabel(f"{name} Feature {feature}")
                    ax.set_title(f"PCA Component {dim} vs {name} Feature {feature}")
                    ax.tick_params(right=True, top=True, which="both", direction="in")
                    pdf.savefig()
                    plt.close()

        # sample checks
        for (proj_data, title) in [(proj[t5s], "PCA Projection: T5s"),
                                   (proj[pls], "PCA Projection: PLSs"),
                                   (proj, "PCA Projection: T5s and PLSs"),
                                   ]:
            print(f"Plotting {title}")
            for norm in [None, colors.LogNorm()]:
                cmin = 0.5
                fig, ax = plt.subplots(figsize=(8, 8))
                _, _, _, im = ax.hist2d(proj_data[:, 0], proj_data[:, 1], bins=100, cmap='gist_heat', cmin=cmin, norm=norm)
                ax.set_xlabel("PCA Component 0")
                ax.set_ylabel("PCA Component 1")
                ax.set_title(title)
                ax.text(1.12, 1.02, "Tracks", transform=ax.transAxes)
                ax.tick_params(right=True, top=True, which="both", direction="in")
                fig.colorbar(im, ax=ax)
                fig.subplots_adjust(right=0.97, left=0.12, bottom=0.09, top=0.95)
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


if __name__ == "__main__":
    main()
