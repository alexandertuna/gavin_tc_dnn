import preprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 16})

READ_FROM_DISK = True
FNAME = "check_fakes.npz"

COLORS = ["red", "blue"]
LABELS = ["Fake", "Real"]
TYPES = {
    4: "T5",
    5: "PT3",
    7: "PT5",
    8: "PLS",
}

def main():

    if READ_FROM_DISK:
        data = np.load(FNAME)
        sims = data["sims"]
        pts = data["pts"]
        etas = data["etas"]
        types = data["types"]

    else:
        # file_path = "../data/LSTNtuple.root"
        file_path = "../data/LSTNtuple.cutNA.root"
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

        for typ in [-1] + list(TYPES.keys()):

            title = "All TCs" if typ == -1 else TYPES[typ]
            real = (sims != -1) & ((typ == -1) | (types == typ))
            fake = (sims == -1) & ((typ == -1) | (types == typ))

            plot_real_and_fake(real=pts[real],
                               fake=pts[fake],
                               bins=np.arange(0.5, 5.0, 0.1),
                               title=title,
                               xlabel=r"$p_{T}$ [GeV]",
                               pdf=pdf)

            plot_real_and_fake(real=etas[real],
                               fake=etas[fake],
                               bins=np.arange(-4, 4.01, 0.1),
                               title=title,
                               xlabel=r"$\eta$",
                               pdf=pdf)
            
            plot_real_and_fake_2d(pts, etas, real, fake, title, pdf)



def plot_real_and_fake(real, fake, bins, title, xlabel, pdf):
    fig, axs = plt.subplots(ncols=2, figsize=(14, 8))
    axs[0].hist([fake, real], bins=bins, label=LABELS, color=COLORS, stacked=True)
    axs[1].hist([fake, real], bins=bins, label=LABELS, color=COLORS, stacked=True,
                weights=get_weights_for_stack_fraction([fake, real], bins))
    axs[1].set_ylim([0, 1.05])
    axs[1].grid()
    axs[0].set_ylabel("Number of tracks")
    axs[1].set_ylabel("Fraction of tracks")
    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.text(0.15, 1.01, title, fontsize=22, transform=ax.transAxes)
        ax.text(0.55, 1.01, "Real", size=22, color=COLORS[1], transform=ax.transAxes)
        ax.text(0.75, 1.01, "Fake", size=22, color=COLORS[0], transform=ax.transAxes)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    pdf.savefig()
    plt.close()


def plot_real_and_fake_2d(pts, etas, real, fake, title, pdf):
    fig, axs = plt.subplots(ncols=2, figsize=(14, 8))
    bins=[np.arange(0.5, 5.0, 0.1),
          np.arange(-4, 4.01, 0.1)]
    extent = bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]
    cmap = "plasma"

    hist_real, _, _ = np.histogram2d(pts[real], etas[real], bins=bins)
    hist_fake, _, _ = np.histogram2d(pts[fake], etas[fake], bins=bins)
    total = hist_real + hist_fake
    total[total == 0] = 1

    im = axs[0].imshow(hist_real.T / total.T, extent=extent, cmap=cmap, aspect='auto', origin='lower')
    axs[0].set_xlabel(r"$p_{T}$ [GeV]")
    axs[0].set_ylabel(r"$\eta$")
    axs[0].set_title(f"{title}: fraction of real tracks")
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(hist_fake.T / total.T, extent=extent, cmap=cmap, aspect='auto', origin='lower')
    axs[1].set_xlabel(r"$p_{T}$ [GeV]")
    axs[1].set_ylabel(r"$\eta$")
    axs[1].set_title(f"{title}: fraction of fake tracks")
    fig.colorbar(im, ax=axs[1])

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.1)
    pdf.savefig()
    plt.close()


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
    ######################################################
    hist_total = hists[0] + hists[1] # np.sum(hists, axis=0)
    ######################################################

    hist_total[hist_total == 0] = 1
    weight_per_bin = 1 / hist_total
    return [make_weights(arr, weight_per_bin, bins) for arr in x]


def make_weights(data, weights_per_bin, bins):
    bin_indices = np.digitize(data, bins) - 1
    # Mask out-of-bounds indices (e.g., right edge)
    valid = (bin_indices >= 0) & (bin_indices < len(weights_per_bin))
    weights = np.zeros_like(data, dtype=float)
    weights[valid] = weights_per_bin[bin_indices[valid]]
    return weights


if __name__ == "__main__":
    main()
