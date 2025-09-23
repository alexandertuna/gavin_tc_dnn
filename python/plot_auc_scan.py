import os
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams["font.size"] = 16

TOPDIR = "/ceph/users/atuna/work/gavin_tc_dnn/experiments/embed_ptetaphi_event_1000_plsQoverPt/run_01"

# run_00
# C_QPTS = [0.25, 0.5, 1.0, 2.0, 4.0]
# C_ETAS = [0.25, 0.5, 1.0, 2.0, 4.0]
# C_PHIS = [0.25, 0.5, 1.0, 2.0, 4.0]

# run_01
C_QPTS = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
C_ETAS = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
C_PHIS = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def main():

    aucs_t5t5_dphys = {}
    for c_qpt in C_QPTS:
        for c_eta in C_ETAS:
            for c_phi in C_PHIS:
                fname = os.path.join(TOPDIR, f"log_{c_qpt}_{c_eta}_{c_phi}.txt")
                exists = os.path.exists(fname)
                if not exists:
                    continue
                [auc_t5t5_demb,
                 auc_t5t5_dr,
                 auc_t5t5_dphys,
                 auc_t5pls_demb,
                 auc_t5pls_dr,
                 auc_t5pls_dphys,
                 ] = get_aucs(fname)
                aucs_t5t5_dphys[c_qpt, c_eta, c_phi] = auc_t5t5_dphys

    print_top(aucs_t5t5_dphys)

    with PdfPages("aucs.pdf") as pdf:
        plot(aucs_t5t5_dphys, pdf)


def print_top(aucs, num=5):
    total = len(aucs)
    sorted_aucs = dict(sorted(aucs.items(), key=lambda item: item[1], reverse=True))
    for it, (key, auc) in enumerate(sorted_aucs.items()):
        if it >= num and it < total - num and key != (1.0, 1.0, 2.0):
            continue
        print(f"{it} {key} {auc:.5f}")

def plot(aucs, pdf):
    vmin, vmax = 0.94, 0.96
    cmap = "Blues"
    all_values = list(aucs.values())
    vmin = min([val for val in all_values if val > 0])
    vmax = max(all_values)

    for c_qpt in C_QPTS:
        fig, ax = plt.subplots(figsize=(8, 8))
        xs, ys, values = [], [], []
        for c_eta in C_ETAS:
            for c_phi in C_PHIS:
                auc = aucs.get((c_qpt, c_eta, c_phi), 0)
                xs.append(c_eta)
                ys.append(c_phi)
                values.append(auc)

        scatter = ax.scatter(xs, ys, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(scatter, ax=ax, label="AUC, T5T5")
        ax.set_xlabel("c_eta")
        ax.set_ylabel("c_phi")
        ax.set_title(f"c_qpt = {c_qpt}")
        # ax.semilogx()
        # ax.semilogy()
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        fig.subplots_adjust(right=0.95, left=0.13, bottom=0.09, top=0.95)
        pdf.savefig()
        plt.close()


def get_aucs(fname):
    aucs = []
    with open(fname, "r") as fi:
        for line in fi:
            if line.startswith("auc"):
                line = line.strip()
                name, auc = line.split(" = ")
                aucs.append(float(auc))
    if len(aucs) != 6:
        return [0]*6
    return aucs


if __name__ == "__main__":
    main()
