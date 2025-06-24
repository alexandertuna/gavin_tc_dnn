import awkward as ak
import uproot
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams["font.size"] = 18

FPATH = Path("pls_t5_embed.root")
BRANCHES = [
    "t5_pt",
    "t5_eta",
    "t5_phi",
    "t5_matched_simIdx",
    "t5_hitIdxs",
    "t5_pMatched",
    "t5_t3_idx0",
    "t5_t3_idx1",
]

XYZ = ["x", "y", "z"]
LAYERS = range(6)
BRANCHES += [f"t5_t3_{layer}_{xyz}" for layer in LAYERS for xyz in XYZ]

def main():
    # print_branches()
    # check_t5_xyz()
    events_and_indices = [
        (470, 5128, 5129),
        (65, 7845, 7847),
        (111, 246, 247),
        (238, 2547, 2553),
        (209, 5671, 5674),
        (44, 5068, 5071),
        (63, 1531, 1537),
        (89, 4767, 4769),
        (431, 3321, 3322),
        (453, 3734, 3737),
        (78, 1212, 1213),
        (142, 6596, 6600),
    ]


    data = get_data()


    for evt, t5_idx0, t5_idx1 in events_and_indices:
        print_info(data, evt, t5_idx0, t5_idx1)


    with PdfPages("test.pdf") as pdf:
        for evt, t5_idx0, t5_idx1 in events_and_indices:
            draw_t5t5s(data, evt, t5_idx0, t5_idx1, pdf)


    with PdfPages("pmatched.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(ak.flatten(data["t5_pMatched"]), bins=100)
        ax.set_xlabel("T5 pMatched")
        ax.set_ylabel("Counts")
        pdf.savefig(fig)
        plt.close()


def print_info(data: ak.Array, evt: int, t5_idx0: int, t5_idx1: int) -> None:

    simIdx_0 = data['t5_matched_simIdx'][evt][t5_idx0]
    simIdx_1 = data['t5_matched_simIdx'][evt][t5_idx1]
    # if ak.almost_equal(simIdx_0, simIdx_1):
    #     raise Exception(f"Confusion! These match: {simIdx_0} vs {simIdx_1}")

    pmatched_0 = data['t5_pMatched'][evt][t5_idx0]
    pmatched_1 = data['t5_pMatched'][evt][t5_idx1]
    print(f"Event {evt}, T5 indices: {t5_idx0} vs {t5_idx1}, percent matched: {pmatched_0:.3f} vs {pmatched_1:.3f}")

    t5_pt = data['t5_pt'][evt]
    t5_eta = data['t5_eta'][evt]
    t5_phi = data['t5_phi'][evt]
    matched_simIdx = data['t5_matched_simIdx'][evt]


def draw_t5t5s(data: ak.Array, evt: int, t5_l: int, t5_r: int, pdf: PdfPages) -> None:

    triplets = [0, 1]
    layers = [0, 1, 2, 3, 4, 5]

    def get_xs_ys(t5):
        xs, ys = [], []
        for triplet in triplets:
            for layer in layers:
                idx = data[f"t5_t3_idx{triplet}"][evt][t5]
                x = data[f"t5_t3_{layer}_x"][evt][idx]
                y = data[f"t5_t3_{layer}_y"][evt][idx]
                print(f"T5 {t5}, triplet {triplet}, layer {layer}: x={x:.4f}, y={y:.4f}")
                xs.append(x)
                ys.append(y)
        return xs, ys

    # get x, y coordinates and pMatched for the two T5s
    xs_l, ys_l = get_xs_ys(t5_l)
    xs_r, ys_r = get_xs_ys(t5_r)
    p_l = data["t5_pMatched"][evt][t5_l]
    p_r = data["t5_pMatched"][evt][t5_r]

    # make plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs_l, ys_l, label=f"T5 {t5_l}", marker="|", s=100, color="blue")
    ax.scatter(xs_r, ys_r, label=f"T5 {t5_r}", marker="_", s=100, color="red")
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"Event {evt}")
    ax.tick_params(right=True, top=True, direction="in")
    ax.text(0.05, 0.88, f"T5 #{t5_l}, P(matched) = {p_l:.3f}", transform=ax.transAxes, fontsize=18, color="blue")
    ax.text(0.05, 0.82, f"T5 #{t5_r}, P(matched) = {p_r:.3f}", transform=ax.transAxes, fontsize=18, color="red")
    fig.subplots_adjust(bottom=0.12, left=0.15, right=0.96, top=0.95)

    # write hits on the plot
    fontsize = 9
    args = {"fontsize": fontsize, "transform": ax.transAxes}
    xcorner, ycorner = 0.60, 0.35
    dx, dy = 0.14, 0.03
    ax.text(xcorner + 0 * dx, ycorner + dy, "x, y", color="blue", **args)
    ax.text(xcorner + 1 * dx, ycorner + dy, "x, y", color="red", **args)
    def draw_hits(xs, ys, xtext, color):
        for it, (x, y) in enumerate(zip(xs, ys)):
            if it in [6, 7]:
                continue
            if it > 7:
                it -= 2
            ax.text(xtext, ycorner - it * dy, f"{x:.2f}, {y:.2f}", color=color, **args)
    draw_hits(xs_l, ys_l, xcorner + 0 * dx, "blue")
    draw_hits(xs_r, ys_r, xcorner + 1 * dx, "red")
    for it in range(len(xs_l)):
        if it in [6, 7]:
            continue
        equal = xs_l[it] == xs_r[it] and ys_l[it] == ys_r[it]
        msg = "Same" if equal else "Diff"
        if it > 7:
            it -= 2
        ax.text(xcorner + 2 * dx, ycorner - it * dy, msg, **args)

    # fin
    pdf.savefig(fig)
    plt.close(fig)


def get_data() -> ak.Array:
    print(f"Reading {FPATH} ...")
    with uproot.open(FPATH) as fi:
        tree = fi["tree"]
        return tree.arrays(BRANCHES)


def print_branches():
    with uproot.open(FPATH) as fi:
        tree = fi["tree"]
        print("Branches in the tree:")
        for branch in tree.keys():
            print(f" - {branch}")


def check_t5_xyz():
    with uproot.open(FPATH) as fi:
        tree = fi["tree"]
        data = tree.arrays(["t5_pt",
                            "t5_t3_idx0",
                            "t5_t3_idx1",
                            "t5_t3_0_x",
                            "t5_t3_0_y",
                            "t5_t3_0_z",
                            "t5_t3_1_x",
                            "t5_t3_1_y",
                            "t5_t3_1_z",
                            "t5_t3_2_x",
                            "t5_t3_2_y",
                            "t5_t3_2_z",
                            "t5_t3_3_x",
                            "t5_t3_3_y",
                            "t5_t3_3_z",
                            "t5_t3_4_x",
                            "t5_t3_4_y",
                            "t5_t3_4_z",
                            "t5_t3_5_x",
                            "t5_t3_5_y",
                            "t5_t3_5_z",
                            "t5_t3_0_phi",
                            "t5_t3_1_phi",
                            "t5_t3_2_phi",
                            "t5_t3_3_phi",
                            "t5_t3_4_phi",
                            "t5_t3_5_phi",
                            "t5_t3_0_layer",
                            "t5_t3_1_layer",
                            "t5_t3_2_layer",
                            "t5_t3_3_layer",
                            "t5_t3_4_layer",
                            "t5_t3_5_layer",
                            ])
    ev = 0
    t5 = 5
    nt5 = len(data["t5_pt"][ev])
    print(f"Number of T5s in event {ev}: {nt5}")
    print(data)
    print(data["t5_pt"])
    print(data["t5_t3_0_x"])
    print(data["t5_pt"][ev])
    print(data["t5_t3_0_x"][ev])
    print(data["t5_t3_idx0"][ev])
    print(data["t5_t3_idx1"][ev])
    print(len(data["t5_pt"][ev]))
    print(len(data["t5_t3_0_x"][ev]))
    print(len(data["t5_t3_idx0"][ev]))
    print(len(data["t5_t3_idx1"][ev]))

    pt = float(data["t5_pt"][ev][t5])
    idx0 = data["t5_t3_idx0"][ev][t5]
    idx1 = data["t5_t3_idx1"][ev][t5]

    # x_0_0 = data["t5_t3_0_x"][ev][idx0]
    # x_0_1 = data["t5_t3_0_x"][ev][idx1]
    # y_0_0 = data["t5_t3_0_y"][ev][idx0]
    # y_0_1 = data["t5_t3_0_y"][ev][idx1]
    # z_0_0 = data["t5_t3_0_z"][ev][idx0]
    # z_0_1 = data["t5_t3_0_z"][ev][idx1]
    # phi_0_0 = data["t5_t3_0_phi"][ev][idx0]
    # phi_0_1 = data["t5_t3_0_phi"][ev][idx1]

    # x_1_0 = data["t5_t3_1_x"][ev][idx0]
    # x_1_1 = data["t5_t3_1_x"][ev][idx1]
    # y_1_0 = data["t5_t3_1_y"][ev][idx0]
    # y_1_1 = data["t5_t3_1_y"][ev][idx1]
    # z_1_0 = data["t5_t3_1_z"][ev][idx0]
    # z_1_1 = data["t5_t3_1_z"][ev][idx1]
    # phi_1_0 = data["t5_t3_1_phi"][ev][idx0]
    # phi_1_1 = data["t5_t3_1_phi"][ev][idx1]

    # xyz_0_0 = f"{x_0_0:8.3f}, {y_0_0:8.3f}, {z_0_0:8.3f}, phi={phi_0_0:6.3f}"
    # xyz_0_1 = f"{x_0_1:8.3f}, {y_0_1:8.3f}, {z_0_1:8.3f}, phi={phi_0_1:6.3f}"
    # xyz_1_0 = f"{x_1_0:8.3f}, {y_1_0:8.3f}, {z_1_0:8.3f}, phi={phi_1_0:6.3f}"
    # xyz_1_1 = f"{x_1_1:8.3f}, {y_1_1:8.3f}, {z_1_1:8.3f}, phi={phi_1_1:6.3f}"
    # print(f"T5 {t5}: {pt=:6.2f}, idx0={idx0}, idx1={idx1}, xyz_0_0=({xyz_0_0}), xyz_0_1=({xyz_0_1})")
    # print(f"T5 {t5}: {pt=:6.2f}, idx0={idx0}, idx1={idx1}, xyz_1_0=({xyz_1_0}), xyz_1_1=({xyz_1_1})")



    idxs = [0, 1]
    vars = ["x", "y", "z", "phi", "layer"]
    layers = [0, 1, 2, 3, 4, 5]

    for t5 in range(1100, 1150):

        coords = {}
        for idx in idxs:
            for var in vars:
                for layer in layers:
                    coords[f"t5_t3_{layer}_{var}_{idx}"] = data[f"t5_t3_{layer}_{var}"][ev][data[f"t5_t3_idx{idx}"][ev][t5]]

        for idx in idxs:
            for layer in layers:
                x = coords[f't5_t3_{layer}_x_{idx}']
                y = coords[f't5_t3_{layer}_y_{idx}']
                z = coords[f't5_t3_{layer}_z_{idx}']
                phi = coords[f't5_t3_{layer}_phi_{idx}']
                global_layer = int(coords[f't5_t3_{layer}_layer_{idx}'])
                varname = f"t5_t3_{layer}_?_{idx}"
                desc = f"{x=:.3f}, {y=:.3f}, {z=:.3f}, {phi=:.3f}, {global_layer=}"
                print(f"T5 {t5}, {varname}: {desc}")
                # print(f"T5 {t5}, layer {layer}: idx={idx}, {desc}")
        print("*" * 50)



if __name__ == "__main__":
    main()
