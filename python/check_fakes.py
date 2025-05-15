import preprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

READ_FROM_DISK = True
FNAME = "check_fakes.npz"

def main():

    if READ_FROM_DISK:
        data = np.load(FNAME)
        sims = data["sims"]
        pts = data["pts"]
        etas = data["etas"]
        types = data["types"]

    else:
        file_path = "../data/LSTNtuple.root"
        # file_path = "../data/LSTNtuple.cutNA.root"
        branches, features, sim_indices, X_left, X_right, y = preprocess.preprocess_data(file_path)

        # flatten everything
        sims  = np.concatenate(sim_indices)
        pts   = np.concatenate([feat[:, 0] for feat in features])
        etas  = np.concatenate([feat[:, 1] for feat in features])
        types = np.concatenate([feat[:, 3] for feat in features])

        # clean up
        sims = np.where(sims == None, -1, sims).astype(int)
        pts = np.power(10, pts)
        etas = etas * 4.0
        types = types * 2.0 + 6.0
        np.savez("check_fakes.npz", sims=sims, pts=pts, etas=etas, types=types)

    # plots
    with PdfPages("check_fakes.pdf") as pdf:

        # plot pt
        pt_fake = pts[sims == -1]
        pt_real = pts[sims != -1]

        fig, axs = plt.subplots(ncols=2, figsize=(14, 8))
        bins = np.arange(0.5, 5.0, 0.1)
        axs[0].hist([pt_fake, pt_real], bins=bins, label=['Fake', 'Real'], stacked=True)

        # counts_fake, _ = np.histogram(pt_fake, bins=bins)
        # counts_real, _ = np.histogram(pt_real, bins=bins)
        # counts_total = counts_fake + counts_real
        # counts_total[counts_total == 0] = 1
        # weights = 1 / counts_total

        # weights_fake = make_weights(pt_fake, weights, bins)
        # weights_real = make_weights(pt_real, weights, bins)
        axs[1].hist([pt_fake, pt_real], bins=bins, label=['Fake', 'Real'], stacked=True,
                    # weights=[weights_fake, weights_real],
                    weights=get_weights_for_stack_fraction([pt_fake, pt_real], bins),
                    )
        axs[1].set_ylim([0, 1.1])
        axs[1].grid()

        pdf.savefig()
        plt.close()

        # plot eta
        eta_fake = etas[sims == -1]
        eta_real = etas[sims != -1]

        fig, axs = plt.subplots(ncols=2, figsize=(14, 8))
        bins = np.arange(-4, 4.01, 0.1)
        axs[0].hist([eta_fake, eta_real], bins=bins, label=['Fake', 'Real'], stacked=True)
        axs[1].hist([eta_fake, eta_real], bins=bins, label=['Fake', 'Real'], stacked=True,
                    weights=get_weights_for_stack_fraction([eta_fake, eta_real], bins))
        axs[1].set_ylim([0, 1.1])
        axs[1].grid()

        pdf.savefig()
        plt.close()



    print("sims.shape", sims.shape)
    print("pts.shape", pts.shape)
    print("etas.shape", etas.shape)
    print("sims[:10]", sims[:10])
    print("n(-1)", np.sum(sims == -1))

    # replace None in sims with -1


    # print("unique sims:", np.unique(sims))


def get_weights_for_stack_fraction(x, bins):
    if not isinstance(x, list):
        raise ValueError("x must be a list of arrays")
    if len(x) == 0:
        raise ValueError("x must not be empty")
    def get_hist(arr, bins):
        hist, _ = np.histogram(arr, bins=bins)
        return hist
    hists = [get_hist(arr, bins) for arr in x]

    # make this nicer plz
    hist_total = hists[0] + hists[1] # np.sum(hists, axis=0)

    hist_total[hist_total == 0] = 1
    weight_per_bin = 1 / hist_total
    return [make_weights(arr, weight_per_bin, bins) for arr in x]

    # counts_total[counts_total == 0] = 1
    # weights = 1 / counts_total
    # weights_fake = make_weights(pt_fake, weights, bins)
    # weights_real = make_weights(pt_real, weights, bins)
    # return [weights_fake, weights_real]

    # bins = np.arange(0.5, 5.0, 0.1)
    # counts_fake, _ = np.histogram(pt_fake, bins=bins)
    # counts_real, _ = np.histogram(pt_real, bins=bins)
    # counts_total = counts_fake + counts_real
    # counts_total[counts_total == 0] = 1
    # weights = 1 / counts_total
    # weights_fake = make_weights(pt_fake, weights, bins)
    # weights_real = make_weights(pt_real, weights, bins)
    # return [weights_fake, weights_real]

def make_weights(data, weights_per_bin, bins):
    bin_indices = np.digitize(data, bins) - 1
    # Mask out-of-bounds indices (e.g., right edge)
    valid = (bin_indices >= 0) & (bin_indices < len(weights_per_bin))
    print("data", data.shape)
    weights = np.zeros_like(data, dtype=float)
    weights[valid] = weights_per_bin[bin_indices[valid]]
    return weights


if __name__ == "__main__":
    main()
