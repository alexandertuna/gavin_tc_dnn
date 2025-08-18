import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from preprocess import load_t5_t5_pairs

def main():

    args = options()
    eta_max = 2.5

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
     true_R_test) = load_t5_t5_pairs(args.pairs_t5t5)

    eta_l = X_left_test[:, 0] * eta_max
    eta_r = X_right_test[:, 0] * eta_max
    mask = eta_l != eta_r

    with PdfPages(args.pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(eta_l[mask] - eta_r[mask], bins=100)
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_xlabel("eta_left - eta_right")
        ax.set_ylabel("Track pairs")
        pdf.savefig()
        ax.semilogy()
        pdf.savefig()
        plt.close()



def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pdf", type=str, default="eta.pdf",
                        help="Path to save the output plots in PDF format")
    parser.add_argument("--pairs_t5t5", type=str, default="pairs_t5t5.pkl",
                        help="Path to the precomputed T5-T5 pairs file")
    parser.add_argument("--pairs_t5pls", type=str, default="pairs_t5pls.pkl",
                        help="Path to the precomputed T5-PLS pairs file")
    parser.add_argument("--pairs_plspls", type=str, default="pairs_plspls.pkl",
                        help="Path to the precomputed PLS-PLS pairs file")
    return parser.parse_args()


if __name__ == "__main__":
    main()
