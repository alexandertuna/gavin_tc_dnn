import argparse
import numpy as np
import pandas as pd
import awkward as ak
import uproot
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams["font.size"] = 16

INVALID_SIM_IDX = -1
BRANCHES = [
    "t5_pt",
    "t5_eta",
    "t5_phi",
    "t5_matched_simIdx",
    "t5_t3_idx0",
    "t5_t3_idx1",
    "t5_t3_0_r",
    "t5_t3_2_r",
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
    "t3_centerX",
    "t3_centerY",
    "pLS_phi",
    "pLS_ptIn",
    "pLS_eta",
    "pLS_deltaPhi",
    "pLS_charge",
    "pLS_matched_simIdx",
]
PARQUET_NAME = "phis.parquet"


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ntuple", type=str, default="/Users/alexandertuna/Downloads/cms/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiCharge.root",
                        help="Input LSTNtuple ROOT file")
    parser.add_argument("--parquet", type=str, default=PARQUET_NAME,
                        help="Path to save the intermediate parquet file")
    parser.add_argument("--pdf", type=str, default="phis.pdf",
                        help="Path to save the output plots in PDF format")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode for additional output")
    parser.add_argument("--load", action="store_true", default=False,
                        help="Load the parquet file instead of processing the ROOT file")
    parser.add_argument("--events", type=int, default=1000,
                        help="Maximum number of events to process")
    return parser.parse_args()


def main() -> None:
    args = options()
    plotter = PhiPlotter(args.events, args.debug)
    if not args.load:
        plotter.load_data(args.ntuple)
        plotter.debug_print_phi_matching()
        plotter.make_track_pair_dataframe()
        # plotter.write_df_to_file(args.parquet)
    # plotter.load_intermediate_data(args.parquet)
    # plotter.plot(args.pdf)


class PhiPlotter:

    def __init__(self, max_ev: int, debug: bool = False):
        self.max_ev = max_ev
        self.debug = debug


    def load_data(self, ntuple_path: str) -> None:
        print(f"Loading data from {ntuple_path}")
        with uproot.open(ntuple_path) as fi:
            tree = fi["tree"]
            if self.debug:
                for key in sorted(list(tree.keys())):
                    print("-", key)
            self.data = tree.arrays(BRANCHES)
            self.n_ev = len(self.data["t5_pt"])
        self.add_branches()


    def add_branches(self) -> None:
        print("Adding branches to data for convenience")

        if self.debug:
            for key in self.data.fields:
                print(f"Key: {key}", self.data[key].type)

        # Flatten simIdx
        for trk in ["t5", "pLS"]:
            self.data[f"{trk}_simIdx"] = ak.firsts(self.data[f"{trk}_matched_simIdx"], axis=-1)
            self.data[f"{trk}_simIdx"] = ak.fill_none(self.data[f"{trk}_simIdx"], INVALID_SIM_IDX)

        # Reconstruct T5 hit phi and r
        idx0s = self.data["t5_t3_idx0"]
        t5_0_phis = self.data["t5_t3_0_phi"][:][idx0s]
        t5_2_phis = self.data["t5_t3_2_phi"][:][idx0s]
        t5_0_rs = self.data["t5_t3_0_r"][:][idx0s]
        t5_2_rs = self.data["t5_t3_2_r"][:][idx0s]
        phi_0_eq = np.abs(t5_0_phis - self.data["t5_phi"]) < 1e-5
        self.data["t5_phi_reco"] = np.where(phi_0_eq, t5_0_phis, t5_2_phis)
        self.data["t5_r"] = np.where(phi_0_eq, t5_0_rs, t5_2_rs)
        almost_equal = ak.almost_equal(self.data["t5_phi"], self.data["t5_phi_reco"], atol=1e-5, dtype_exact=False)
        if not np.all(almost_equal):
            raise ValueError("Mismatch in t5_phi and derived t5_phi")

        # Reconstruct T5 charge


    def make_track_pair_dataframe(self) -> None:
        print("Creating track pair dataframe")
        dfs = []
        for i_ev in range(self.n_ev):
            if i_ev >= self.max_ev:
                break
            dfs.append( self.make_track_pair_dataframe_one_event(i_ev) )

        print(f"Concatenating event-by-event dataframes")
        self.df = pd.concat(dfs, ignore_index=True)

        print(f"Total track pairs found: {len(self.df)}")
        self.add_columns()
        if self.debug:
            print("First 10 pairs:\n", self.df.head(10))


    def add_columns(self) -> None:
        print("Adding additional columns to the dataframe")
        print("t5_phi", self.df["t5_phi"].min(), self.df["t5_phi"].max())
        print(self.df["t5_phi"])
        self.df["pLS_phi_p_deltaPhi"] = normalize_angle(self.df["pLS_phi"] + self.df["pLS_deltaPhi"])
        self.df["pLS_phi_m_deltaPhi"] = normalize_angle(self.df["pLS_phi"] - self.df["pLS_deltaPhi"])
        self.df["dphi_t5_pLS"] = np.abs(normalize_angle(self.df["t5_phi"] - self.df["pLS_phi"]))
        self.df["dphi_t5_pLS_p_deltaPhi"] = np.abs(normalize_angle(self.df["t5_phi"] - self.df["pLS_phi_p_deltaPhi"]))
        self.df["dphi_t5_pLS_m_deltaPhi"] = np.abs(normalize_angle(self.df["t5_phi"] - self.df["pLS_phi_m_deltaPhi"]))
        print(self.df.head(10))
        print(self.df["dphi_t5_pLS"].min(), self.df["dphi_t5_pLS"].max())

        # corrected pls phi
        # correction based on the B*c and rxy(pls)-rxy(T5):
        #   it's roughly delta = dr/173/pt*charge
        # t5_t3_0_r, t5_t3_1_r
        # pixel barrel r_max is about 15
        self.df["pLS_r"] = 15.0
        self.df["t5_r"] = 36.0 # , 51.0
        self.df["dphi_prop"] = (self.df["t5_r"] - self.df["pLS_r"]) / 10.0 / 173.0 / self.df["pLS_ptIn"] * self.df["pLS_charge"]
        self.df["pLS_phi_corr_m"] = normalize_angle(self.df["pLS_phi_m_deltaPhi"] - self.df["dphi_prop"])
        self.df["pLS_phi_corr_p"] = normalize_angle(self.df["pLS_phi_m_deltaPhi"] + self.df["dphi_prop"])
        self.df["pLS_phi_corr_p2"] = normalize_angle(self.df["pLS_phi_m_deltaPhi"] + 0.5*self.df["dphi_prop"])



    def make_track_pair_dataframe_one_event(self, i_ev: int) -> pd.DataFrame:
        t5_idx, pls_idx = self.match_tracks_one_event(i_ev)
        return pd.DataFrame({
            "event": i_ev,
            "t5_pt": self.data["t5_pt"][i_ev][t5_idx],
            "t5_eta": self.data["t5_eta"][i_ev][t5_idx],
            "t5_phi": self.data["t5_phi"][i_ev][t5_idx],
            "t5_r": self.data["t5_r"][i_ev][t5_idx],
            "pLS_phi": self.data["pLS_phi"][i_ev][pls_idx],
            "pLS_ptIn": self.data["pLS_ptIn"][i_ev][pls_idx],
            "pLS_eta": self.data["pLS_eta"][i_ev][pls_idx],
            "pLS_charge": self.data["pLS_charge"][i_ev][pls_idx],
            "pLS_deltaPhi": self.data["pLS_deltaPhi"][i_ev][pls_idx],
            "t5_simIdx": self.data["t5_simIdx"][i_ev][t5_idx],
            "pLS_simIdx": self.data["pLS_simIdx"][i_ev][pls_idx],
            "t5_idx": t5_idx,
            "pLS_idx": pls_idx
        })


    def match_tracks_one_event(self, i_ev: int) -> np.ndarray:
        t5_simIdx = self.data["t5_simIdx"][i_ev]
        pls_simIdx = self.data["pLS_simIdx"][i_ev]

        # count the number of tracks
        n_t, n_p = len(t5_simIdx), len(pls_simIdx)
        if n_p == 0 or n_t == 0:
            raise ValueError(f"No tracks in event {i_ev}")

        # make all possible pairs (i, j)
        idx_p, idx_t = np.indices( (n_p, n_t) )
        idx_p, idx_t = idx_p.flatten(), idx_t.flatten()

        # compare sim indices
        simidx_p = pls_simIdx[idx_p]
        simidx_t = t5_simIdx[idx_t]
        simidx_same = (simidx_p == simidx_t) & (simidx_p != INVALID_SIM_IDX)
        print(f"Event {i_ev:04}: {n_t=}, {n_p=} found {np.sum(simidx_same)} / {len(simidx_same)} matching pairs")

        if self.debug and i_ev == 0:
            print(f"Debug info for event {i_ev}:")
            print("Matching t5 indices:", idx_t[simidx_same])
            print("Matching pls indices:", idx_p[simidx_same])
            print("Matching t5 sim indices:", t5_simIdx[idx_t[simidx_same]])
            print("Matching pls sim indices:", pls_simIdx[idx_p[simidx_same]])

        return idx_t[simidx_same], idx_p[simidx_same]


    def write_df_to_file(self, filename: str) -> None:
        print(f"Writing dataframe to {filename}")
        self.df.to_parquet(filename)


    def load_intermediate_data(self, filename: str) -> None:
        print(f"Loading intermediate data from {filename}")
        self.df = pd.read_parquet(filename)
        if self.debug:
            print("Dataframe loaded with shape:", self.df.shape)
            print("Columns:", self.df.columns.tolist())


    def plot(self, pdf_path: str) -> None:
        print(f"Plotting results to {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            self.plot_pt_eta(pdf)
            #self.plot_phis(pdf)
            self.plot_dphis(pdf)
            self.plot_dphi_vs_pt(pdf)
            #self.plot_dphi_vs_pt_regions(pdf)
            #self.plot_publicity_dphi(pdf)


    def plot_pt_eta(self, pdf: PdfPages) -> None:
        print("Plotting pt/eta distributions")
        quantities = [
            "t5_pt",
            "t5_eta",
            "pLS_ptIn",
            "pLS_eta",
        ]
        fig, axs = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
        for i, quant in enumerate(quantities):
            is_pt = "_pt" in quant
            bins = np.arange(0, 10, 0.1) if is_pt else np.arange(-2.6, 2.7, 0.04)
            ax = axs[i // 2, i % 2]
            ax.hist(self.df[quant], bins=bins, label=quant, color=f"C{i}")
            ax.set_xlabel(quant)
            ax.set_ylabel("Counts")
            if is_pt:
                ax.semilogy()
            fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.1, hspace=0.2, wspace=0.4)
        pdf.savefig()
        plt.close()


    def plot_phis(self, pdf: PdfPages) -> None:
        print("Plotting phi distributions")
        phis = [
            "t5_phi",
            "pLS_phi",
            "pLS_phi_m_deltaPhi",
            "pLS_phi_p_deltaPhi",
        ]
        fig, axs = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
        for i, phi in enumerate(phis):
            ax = axs[i // 2, i % 2]
            ax.hist(self.df[phi], bins=np.arange(-3.2, 3.3, 0.1), label=phi, color=f"C{i}")
            ax.set_xlabel("Phi (radians)")
            ax.set_ylabel("Counts")
            ax.set_title(f"Distribution of {phi}")
            ax.legend()
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
        pdf.savefig()
        plt.close()


    def plot_dphis(self, pdf: PdfPages) -> None:
        print("Plotting delta phi distributions")
        pls_phis = [
            "pLS_phi_p_deltaPhi",
            "pLS_phi_m_deltaPhi",
            "pLS_phi",
            "pLS_phi_corr_p",
            "pLS_phi_corr_p2",
            "pLS_phi_corr_m",
        ]
        # 2D diff
        for i, pls_phi in enumerate(pls_phis):
            fig, ax = plt.subplots(figsize=(8, 8))
            _, _, _, im = ax.hist2d(self.df["t5_phi"], self.df[pls_phi], bins=np.arange(-3.2, 3.3, 0.05), label=pls_phi, cmin=0.5, cmap="hot")
            ax.set_xlabel("T5 phi (radians)")
            ax.set_ylabel("pLS phi (radians)")
            ax.set_title(f"Distribution of {pls_phi}")
            fig.colorbar(im, ax=ax) # , pad=self.pad
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
            pdf.savefig()
            plt.close()

        # 1D diff
        bins = np.arange(-0.25, 0.25, 0.002)
        fig, ax = plt.subplots(figsize=(8, 8))
        for pls_phi in pls_phis:
            mask = np.abs(self.df["t5_eta"]) < 0.5
            ax.hist(normalize_angle(self.df["t5_phi"] - self.df[pls_phi])[mask],
                    bins=bins,
                    label=pls_phi,
                    alpha=0.5,
                    histtype='stepfilled',
                    edgecolor='black',
                    linewidth=0.5)
        ax.set_xlabel("dphi (radians)")
        ax.set_ylabel("Counts")
        ax.set_title("Distribution of dphi")
        ax.legend()
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
        pdf.savefig()
        plt.close()


    def plot_dphi_vs_pt(self, pdf: PdfPages) -> None:
        pls_phis = [
            "pLS_phi_p_deltaPhi",
            "pLS_phi_m_deltaPhi",
            "pLS_phi",
            "pLS_phi_corr_p2",
            "pLS_phi_corr_p",
            "pLS_phi_corr_m",
        ]
        for pls_phi in pls_phis:
            fig, ax = plt.subplots(figsize=(8, 8))

            x = 1 / self.df["t5_pt"]
            y = normalize_angle(self.df["t5_phi"] - self.df[pls_phi])
            mask = np.abs(self.df["t5_eta"]) < 0.5

            _, _, _, im = ax.hist2d(x[mask],
                                    y[mask],
                                    bins=[
                                        np.arange(0, 1.5, 0.01), 
                                        np.arange(-0.25, 0.25, 0.005)
                                    ],
                                    cmin=0.5,
                                    cmap="hsv",
                                    )
            ax.set_xlabel("T5 pT")
            ax.set_ylabel(f"T5 phi - {pls_phi} (radians)")
            ax.set_title(f"Distribution of dphi vs pT ({pls_phi})")
            fig.colorbar(im, ax=ax) # , pad=self.pad
            fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.08)
            pdf.savefig()
            plt.close()


    def plot_dphi_vs_pt_regions(self, pdf: PdfPages) -> None:
        pls_phis = [
            "pLS_phi_p_deltaPhi",
            "pLS_phi_m_deltaPhi",
            "pLS_phi",
        ]
        pad = 0.01
        bins = [
            np.arange(0, 1.5, 0.01),
            np.arange(-0.25, 0.25, 0.005)
        ]
        for pls_phi in pls_phis:
            for charge in [-1, 1]:
                for eta_lo, eta_hi in [
                    (0.0, 2.5),
                    (0.0, 1.0),
                    (2.0, 2.5),
                ]:
                    mask = (self.df["pLS_charge"] == charge) \
                        & (np.abs(self.df["pLS_eta"]) >= eta_lo) \
                        & (np.abs(self.df["pLS_eta"]) < eta_hi)
                    print(f"Plotting for {charge=}, {eta_lo=}, {eta_hi=}, {np.sum(mask)} events")
                    x = 1/self.df["t5_pt"][mask]
                    y = normalize_angle(self.df["t5_phi"][mask] - self.df[pls_phi][mask])
                    fig, ax = plt.subplots(figsize=(8, 8))
                    _, _, _, im = ax.hist2d(x, y,
                                            bins=bins,
                                            cmin=0.5,
                                            cmap="hsv",
                                            )
                    ax.set_xlabel("T5 pT")
                    ax.set_ylabel(f"T5 phi - {pls_phi} (radians)")
                    ax.set_title(f"{pls_phi}, {eta_lo} < eta < {eta_hi}, q={charge}")
                    ax.text(1.08, 1.01, f"Pairs", transform=ax.transAxes)
                    fig.colorbar(im, ax=ax, pad=pad)
                    fig.subplots_adjust(left=0.15, right=0.99, top=0.95, bottom=0.08)
                    pdf.savefig()
                    plt.close()


    def plot_publicity_dphi(self, pdf: PdfPages) -> None:
        print("Plotting delta phi distributions")
        args = {
            "bins": np.arange(-0.25, 0.25, 0.002),
            "alpha": 0.75,
            "histtype": "stepfilled",
            "edgecolor": "black",
            "linewidth": 2.0
        }
        label = {
            "pLS_phi_m_deltaPhi": r"$\phi^{T5}_{r} - \phi^{pLS}_{r}$",
            "pLS_phi": r"$\phi^{T5}_{r} - \phi^{pLS}_{p}$",
        }
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(normalize_angle(self.df["t5_phi"] - self.df["pLS_phi_m_deltaPhi"]), **args, label=label["pLS_phi_m_deltaPhi"])
        ax.hist(normalize_angle(self.df["t5_phi"] - self.df["pLS_phi"]), **args, label=label["pLS_phi"])
        ax.set_xlabel(r"$\Delta\phi$(pLS, T5) [radians]")
        ax.set_ylabel("Pairs of T5/pLS tracks")
        ax.set_title(r"T5 and pLS tracks matched by sim. parent")
        ax.legend(frameon=False, loc="upper right")
        ax.text(0.79, 0.83, "(default)", transform=ax.transAxes) # , fontsize=14)
        fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
        pdf.savefig()
        plt.close()


    def debug_print_phi_matching(self) -> None:

        if not self.debug:
            return

        def check():
            idx0s = self.data["t5_t3_idx0"]
            t5_t3_0_phis = self.data["t5_t3_0_phi"][:][idx0s]
            t5_t3_2_phis = self.data["t5_t3_2_phi"][:][idx0s]
            t5_phis = np.where(np.abs(t5_t3_0_phis - self.data["t5_phi"]) < 1e-5, t5_t3_0_phis, t5_t3_2_phis)
            almost_equal = ak.almost_equal(self.data["t5_phi"], t5_phis, atol=1e-5, dtype_exact=False)
            equal = np.abs(self.data["t5_phi"] - t5_phis) < 1e-5
            print(f"All events: {almost_equal=}. {np.sum(equal)} / {len(ak.flatten(t5_phis))} T5 tracks have matching phi")
        check()

        # Table header
        names = ["t5_phi", "my_phi",
                 "t5_t3_0_phi", "t5_t3_1_phi", "t5_t3_2_phi", "t5_t3_3_phi", "t5_t3_4_phi", "t5_t3_5_phi",
                 "t5_t3_0_phi", "t5_t3_1_phi", "t5_t3_2_phi", "t5_t3_3_phi", "t5_t3_4_phi", "t5_t3_5_phi",
                 ]
        print(" "*6, " ".join(f"{name:>12}" for name in names), "layer")

        # Table rows
        for ev in range(10):
            n_t5 = len(self.data["t5_pt"][ev])
            subset = 10

            idx0s = self.data["t5_t3_idx0"][ev]
            t5_t3_0_phis = self.data["t5_t3_0_phi"][ev][idx0s]
            t5_t3_2_phis = self.data["t5_t3_2_phi"][ev][idx0s]
            t5_t3_0_rs = self.data["t5_t3_0_r"][ev][idx0s]
            t5_t3_2_rs = self.data["t5_t3_2_r"][ev][idx0s]

            # set t5_r equal to t5_t3_0_r if t5_phi is equal to t5_t3_0_phi. otherwise, set it to t5_t3_2_r
            t5_phis = np.where(np.abs(t5_t3_0_phis - self.data["t5_phi"][ev]) < 1e-5, t5_t3_0_phis, t5_t3_2_phis)
            t5_rs = np.where(np.abs(t5_t3_0_phis - self.data["t5_phi"][ev]) < 1e-5, t5_t3_0_rs, t5_t3_2_rs)

            almost_equal = ak.almost_equal(self.data["t5_phi"][ev], t5_phis, atol=1e-5, dtype_exact=False)
            equal = np.abs(self.data["t5_phi"][ev] - t5_phis) < 1e-5
            print(f"Event {ev:04d}: {almost_equal=}. {np.sum(equal)} / {n_t5} T5 tracks have matching phi")

            if ev > 0:
                continue

            for i in np.random.choice(range(n_t5), size=subset):
                idx0 = self.data["t5_t3_idx0"][ev][i]
                idx1 = self.data["t5_t3_idx1"][ev][i]
                the_phi = self.data["t5_phi"][ev][i]
                my_phi = t5_phis[i]
                phis = [
                    self.data["t5_t3_0_phi"][ev][idx0],
                    self.data["t5_t3_1_phi"][ev][idx0],
                    self.data["t5_t3_2_phi"][ev][idx0],
                    self.data["t5_t3_3_phi"][ev][idx0],
                    self.data["t5_t3_4_phi"][ev][idx0],
                    self.data["t5_t3_5_phi"][ev][idx0],
                    self.data["t5_t3_0_phi"][ev][idx1],
                    self.data["t5_t3_1_phi"][ev][idx1],
                    self.data["t5_t3_2_phi"][ev][idx1],
                    self.data["t5_t3_3_phi"][ev][idx1],
                    self.data["t5_t3_4_phi"][ev][idx1],
                    self.data["t5_t3_5_phi"][ev][idx1],
                ]
                layers = [
                    self.data["t5_t3_0_layer"][ev][idx0],
                    self.data["t5_t3_1_layer"][ev][idx0],
                    self.data["t5_t3_2_layer"][ev][idx0],
                    self.data["t5_t3_3_layer"][ev][idx0],
                    self.data["t5_t3_4_layer"][ev][idx0],
                    self.data["t5_t3_5_layer"][ev][idx0],
                    self.data["t5_t3_0_layer"][ev][idx1],
                    self.data["t5_t3_1_layer"][ev][idx1],
                    self.data["t5_t3_2_layer"][ev][idx1],
                    self.data["t5_t3_3_layer"][ev][idx1],
                    self.data["t5_t3_4_layer"][ev][idx1],
                    self.data["t5_t3_5_layer"][ev][idx1],
                ]
                # print([int(x) for x in layers])
                # print([f"{x:08x}" for x in moduleTypes])
                matching = ["*" if abs(phi-the_phi) < 1e-5 else " " for phi in phis]
                for im, match in enumerate(matching):
                    if match == "*":
                        layer = layers[im]

                print(f"{i:5d}: "
                    f"{the_phi:11.5f} "
                    f"{my_phi:11.5f} "
                    f"{phis[0]:11.5f}{matching[0]} "
                    f"{phis[1]:11.5f}{matching[1]} "
                    f"{phis[2]:11.5f}{matching[2]} "
                    f"{phis[3]:11.5f}{matching[3]} "
                    f"{phis[4]:11.5f}{matching[4]} "
                    f"{phis[5]:11.5f}{matching[5]} "
                    f"{phis[6]:11.5f}{matching[6]} "
                    f"{phis[7]:11.5f}{matching[7]} "
                    f"{phis[8]:11.5f}{matching[8]} "
                    f"{phis[9]:11.5f}{matching[9]} "
                    f"{phis[10]:11.5f}{matching[10]} "
                    f"{phis[11]:11.5f}{matching[11]} ",
                    f"{layer:5}")

                if matching.count("*") not in [1, 2]:
                    raise Exception(f"Expected exactly one matching phi, found {matching.count('*')} for event {ev}, track {i}")
                if matching[0] != "*" and matching[2] != "*":
                    raise Exception(f"Expected the first or third phi to match, found none for event {ev}, track {i}")



def normalize_angle(delta):
    # Adjust delta to be within the range [-pi, pi]
    delta[delta > np.pi] -= 2 * np.pi
    delta[delta < -np.pi] += 2 * np.pi
    return delta


if __name__ == "__main__":
    main()
