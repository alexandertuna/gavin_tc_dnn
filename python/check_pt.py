import awkward as ak
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FNAME = "../data/LSTNtuple.cutNA.root"
# FNAME = "../data/LSTNtuple.root"

plt.rcParams.update({'font.size': 20})

def main():
    with uproot.open(f"{FNAME}:tree") as tr:
        print(tr)
        data = tr.arrays(["tc_pt",
                          "tc_type"])

    # transform the data
    pts = ak.flatten(data["tc_pt"])
    types = ak.flatten(data["tc_type"])
    unique_types = np.unique(ak.flatten(data["tc_type"]))
    pts_per_type = [
        pts[types == ty]
        for ty in unique_types
    ]
    for it, ty in enumerate(unique_types):
        print(f"Type {ty}: {len(pts_per_type[it])} tracks")

    # plot mise en place
    labels = [type_to_name(ty) for ty in unique_types]
    colors = list(mcolors.TABLEAU_COLORS.values())
    colors = colors[:len(unique_types)]

    # make plots
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(pts_per_type,
            stacked=True,
            color=colors,
            label=labels,
            bins=np.arange(0.5, 1.5, 0.01),
            )

    # beautify and save
    for it, (color, label) in enumerate(zip(reversed(colors), reversed(labels))):
       ax.text(0.1, 0.9-it*0.05, label, color=color, transform=ax.transAxes)
    ax.set_xlabel("Track candidate $p_{T}$ [GeV] (tc_pt)")
    ax.set_ylabel("Counts")
    ax.semilogy()
    ax.tick_params(top=True)
    ax.tick_params(right=True, which="minor")
    fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.10)
    fig.savefig("tc_pt.pdf")


def type_to_name(typ: int) -> str:
    if typ == 4:
        return 'PT5'
    elif typ == 5:
        return 'PT3'
    elif typ == 7:
        return 'T5'
    elif typ == 8:
        return 'PLS'
    else:
        raise ValueError(f"Unknown type: {typ}")


if __name__ == "__main__":
    main()
