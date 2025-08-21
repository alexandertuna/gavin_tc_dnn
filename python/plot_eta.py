import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams["font.size"] = 16

from preprocess import load_t5_t5_pairs, load_t5_pls_pairs, load_pls_pls_pairs

ETA_MAX_T5 = 2.5
ETA_MAX_PLS = 4.0


def main():

    args = options()

    [eta_l_sim, eta_r_sim, eta_l_dis, eta_r_dis,
     phi_l_sim, phi_r_sim, phi_l_dis, phi_r_dis] = get_t5_t5_info(args.pairs_t5t5)

    [eta_t_sim, eta_p_sim, eta_t_dis, eta_p_dis,
     phi_t_sim, phi_p_sim, phi_t_dis, phi_p_dis] = get_t5_pls_info(args.pairs_t5pls)

    [eta_pl_sim, eta_pr_sim, eta_pl_dis, eta_pr_dis,
     phi_pl_sim, phi_pr_sim, phi_pl_dis, phi_pr_dis] = get_pls_pls_info(args.pairs_plspls)

    plots = [
        (eta_l_sim, eta_r_sim, "T5-T5 duplicates", "eta_l - eta_r"),
        (eta_l_dis, eta_r_dis, "T5-T5 non-duplicates", "eta_l - eta_r"),
        (phi_l_sim, phi_r_sim, "T5-T5 duplicates", "phi_l - phi_r"),
        (phi_l_dis, phi_r_dis, "T5-T5 non-duplicates", "phi_l - phi_r"),
        (eta_t_sim, eta_p_sim, "T5-PLS duplicates", "eta_t5 - eta_pls"),
        (eta_t_dis, eta_p_dis, "T5-PLS non-duplicates", "eta_t5 - eta_pls"),
        (phi_t_sim, phi_p_sim, "T5-PLS duplicates", "phi_t5 - phi_pls"),
        (phi_t_dis, phi_p_dis, "T5-PLS non-duplicates", "phi_t5 - phi_pls"),
        (eta_pl_sim, eta_pr_sim, "PLS-PLS duplicates", "eta_pls_l - eta_pls_r"),
        (eta_pl_dis, eta_pr_dis, "PLS-PLS non-duplicates", "eta_pls_l - eta_pls_r"),
        (phi_pl_sim, phi_pr_sim, "PLS-PLS duplicates", "phi_pls_l - phi_pls_r"),
        (phi_pl_dis, phi_pr_dis, "PLS-PLS non-duplicates", "phi_pls_l - phi_pls_r"),
    ]


    with PdfPages(args.pdf) as pdf:

        for (left, right, title, xlabel) in plots:

            diff = left - right
            if "phi" in xlabel:
                diff[diff > np.pi] -= 2 * np.pi
                diff[diff < -np.pi] += 2 * np.pi

            identical = np.abs(diff) < 1e-6
            frac_excluded = identical.sum() / len(identical)
            percent = int(100*frac_excluded)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(diff[~identical], bins=100)
            ax.set_axisbelow(True)
            ax.grid()
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Track pairs")
            ax.set_title(f"{title}, excluding {percent}%")
            fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.10)
            pdf.savefig()
            # ax.semilogy()
            # pdf.savefig()
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


def get_t5_t5_info(pairs: str):

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
     true_R_test) = load_t5_t5_pairs(pairs)

    sim = (y_t5_test == 0)
    dis = (y_t5_test == 1)

    eta_l = X_left_test[:, 0] * ETA_MAX_T5
    eta_r = X_right_test[:, 0] * ETA_MAX_T5

    phi_l = np.arctan2(X_left_test[:, 2], X_left_test[:, 1])
    phi_r = np.arctan2(X_right_test[:, 2], X_right_test[:, 1])

    return [
        eta_l[sim], eta_r[sim],
        eta_l[dis], eta_r[dis],
        phi_l[sim], phi_r[sim],
        phi_l[dis], phi_r[dis],
    ]


def get_t5_pls_info(pairs: str):

    (X_pls_train,
     X_pls_test,
     X_t5raw_train,
     X_t5raw_test,
     y_pls_train,
     y_pls_test,
     w_pls_train,
     w_pls_test) = load_t5_pls_pairs(pairs)

    sim = (y_pls_test == 0)
    dis = (y_pls_test == 1)

    eta_t = X_t5raw_test[:, 0] * ETA_MAX_T5
    eta_p = X_pls_test[:, 0] * ETA_MAX_PLS

    phi_t = np.arctan2(X_t5raw_test[:, 2], X_t5raw_test[:, 1])
    phi_p = np.arctan2(X_pls_test[:, 3], X_pls_test[:, 2])

    return [
        eta_t[sim], eta_p[sim],
        eta_t[dis], eta_p[dis],
        phi_t[sim], phi_p[sim],
        phi_t[dis], phi_p[dis],
    ]


def get_pls_pls_info(pairs: str):
    (X_pls_left,
     X_pls_right,
     y_pls) = load_pls_pls_pairs(pairs)

    sim = (y_pls == 0)
    dis = (y_pls == 1)

    eta_l = X_pls_left[:, 0] * ETA_MAX_PLS
    eta_r = X_pls_right[:, 0] * ETA_MAX_PLS

    phi_l = np.arctan2(X_pls_left[:, 3], X_pls_left[:, 2])
    phi_r = np.arctan2(X_pls_right[:, 3], X_pls_right[:, 2])

    return [
        eta_l[sim], eta_r[sim],
        eta_l[dis], eta_r[dis],
        phi_l[sim], phi_r[sim],
        phi_l[dis], phi_r[dis],
    ]


if __name__ == "__main__":
    main()
