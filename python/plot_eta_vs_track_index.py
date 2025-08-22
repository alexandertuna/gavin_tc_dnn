import uproot
import awkward as ak
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams["font.size"] = 16

BRANCHES = [
    "t5_eta",
    "t5_phi",
]
NTUPLE_PATH = "/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiChargeXYZ.root"
SCATTER = True  

def main():

    with uproot.open(NTUPLE_PATH) as fi:
        tree = fi["tree"]
        data = tree.arrays(BRANCHES)

    print(data["t5_eta"])
    print(data["t5_phi"])

    evt = 0
    eta = data["t5_eta"][evt].to_numpy()
    idx = np.arange(len(eta))

    with PdfPages("eta_vs_track_index.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8, 8))
        if SCATTER:
            ax.scatter(idx, eta, s=1, color="blue")
        else:
            _, _, _, im = ax.hist2d(idx, eta, bins=200, cmin=0.5)
        ax.set_xlabel("T5 array index")
        ax.set_ylabel("T5 eta")
        ax.set_title(f"ttbar, PU200, event {evt}")
        ax.grid()
        if SCATTER:
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.97, top=0.94)
        else:
            fig.colorbar(im, ax=ax, pad=0.01)
            fig.subplots_adjust(bottom=0.08, left=0.13, right=0.95, top=0.94)
        pdf.savefig()
        plt.close()


if __name__ == "__main__":
    main()
