import awkward as ak
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FNAME = "../data/LSTNtuple.cutNA.root"

plt.rcParams.update({'font.size': 20})

def main():
    with uproot.open(f"{FNAME}:tree") as tr:
        print(tr)
        data = tr.arrays(["tc_pt"])

    fig, ax = plt.subplots(figsize=(8, 8))
    print(ak.flatten(data["tc_pt"]))
    ax.hist(ak.flatten(data["tc_pt"]), bins=np.arange(0.5, 1.5, 0.01))
    ax.set_xlabel("Track candidate $p_{T}$ [GeV] (tc_pt)")
    ax.set_ylabel("Counts")
    ax.semilogy()
    ax.tick_params(right=True, top=True)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.10)
    fig.savefig("tc_pt.pdf")


if __name__ == "__main__":
    main()
