import argparse
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors

from preprocess import load_t5_features, load_pls_features
from preprocess import load_t5_t5_pairs, load_t5_pls_pairs
from ml import EmbeddingNetT5, EmbeddingNetpLS

BONUS_FEATURES = 2

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="model_weights.pth",
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
    parser.add_argument("--n_pca", type=int, default=6,
                        help="Number of PCA components")
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
    checkpoint = torch.load(model_weights)
    embed_t5.load_state_dict(checkpoint["embed_t5"])
    embed_pls.load_state_dict(checkpoint["embed_pls"])
    embed_t5.eval()
    embed_pls.eval()


    print("Embedding T5s and PLSs")
    x_t5 = torch.tensor(X_left_test[:, :-BONUS_FEATURES], requires_grad=False)
    x_pls = torch.tensor(X_pls_test[:, :-BONUS_FEATURES], requires_grad=False)
    embedded_t5 = embed_t5(x_t5).detach().numpy()
    embedded_pls = embed_pls(x_pls).detach().numpy()
    print(f"Embedded T5s shape: {embedded_t5.shape}")
    print(f"Embedded PLSs shape: {embedded_pls.shape}")


    print("Performing PCA on embedded T5s and PLSs")
    pca_t5 = PCA(n_components=args.n_pca)
    pca_pls = PCA(n_components=args.n_pca)
    proj_t5 = pca_t5.fit_transform(embedded_t5)
    proj_pls = pca_pls.fit_transform(embedded_pls)
    print(f"PCA T5 projection shape: {proj_t5.shape}")
    print(f"PCA PLS projection shape: {proj_pls.shape}")

    with PdfPages(pdf_name) as pdf:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist2d(proj_t5[:, 0], proj_t5[:, 1], bins=100, cmap='Blues')
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title("PCA Projection of T5s")
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist2d(proj_t5[:, 0], proj_t5[:, 1], bins=100, cmap='Blues', norm=colors.LogNorm())
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title("PCA Projection of T5s")
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
