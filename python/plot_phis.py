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
# mpl.rcParams['text.usetex'] = True

INVALID_SIM_IDX = -1
BRANCHES = [
    "t5_pt",
    "t5_eta",
    "t5_phi",
    "t5_matched_simIdx",
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
        plotter.make_track_pair_dataframe()
        plotter.write_df_to_file(args.parquet)
    plotter.load_intermediate_data(args.parquet)
    plotter.plot(args.pdf)


class PhiPlotter:

    def __init__(self, max_ev: int, debug: bool = False):
        self.max_ev = max_ev
        self.debug = debug


    def load_data(self, ntuple_path: str) -> None:
        print(f"Loading data from {ntuple_path}")
        with uproot.open(ntuple_path) as fi:
            tree = fi["tree"]
            self.data = tree.arrays(BRANCHES)
            self.n_ev = len(self.data["t5_pt"])
        self.add_branches()


    def add_branches(self) -> None:
        print("Adding branches to data for convenience")
        for trk in ["t5", "pLS"]:
            self.data[f"{trk}_simIdx"] = ak.firsts(self.data[f"{trk}_matched_simIdx"], axis=-1)
            self.data[f"{trk}_simIdx"] = ak.fill_none(self.data[f"{trk}_simIdx"], INVALID_SIM_IDX)
        if self.debug:
            for key in self.data.fields:
                print(f"Key: {key}", self.data[key].type)


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


    def make_track_pair_dataframe_one_event(self, i_ev: int) -> pd.DataFrame:
        t5_idx, pls_idx = self.match_tracks_one_event(i_ev)
        return pd.DataFrame({
            "event": i_ev,
            "t5_pt": self.data["t5_pt"][i_ev][t5_idx],
            "t5_eta": self.data["t5_eta"][i_ev][t5_idx],
            "t5_phi": self.data["t5_phi"][i_ev][t5_idx],
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
            #self.plot_phis(pdf)
            self.plot_dphis(pdf)
            #self.plot_dphi_vs_pt(pdf)
            #self.plot_dphi_vs_pt_regions(pdf)
            #self.plot_publicity_dphi(pdf)


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
            ax.hist(normalize_angle(self.df["t5_phi"] - self.df[pls_phi]), bins=bins, label=pls_phi, alpha=0.5, histtype='stepfilled', edgecolor='black', linewidth=0.5)
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
        ]
        for pls_phi in pls_phis:
            fig, ax = plt.subplots(figsize=(8, 8))
            _, _, _, im = ax.hist2d(1/self.df["t5_pt"],
                                    normalize_angle(self.df["t5_phi"] - self.df[pls_phi]),
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



def normalize_angle(delta):
    # Adjust delta to be within the range [-pi, pi]
    delta[delta > np.pi] -= 2 * np.pi
    delta[delta < -np.pi] += 2 * np.pi
    return delta


if __name__ == "__main__":
    main()
