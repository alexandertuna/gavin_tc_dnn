import argparse
import numpy as np
import pandas as pd
import awkward as ak
import uproot
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INVALID_SIM_IDX = -1
BRANCHES = [
    "t5_pt",
    "t5_eta",
    "t5_phi",
    "t5_matched_simIdx",
    "pLS_ptIn",
    "pLS_eta",
    "pLS_phi",
    "pLS_deltaPhi",
    "pLS_matched_simIdx",
]


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ntuple", type=str, default="/Users/alexandertuna/Downloads/cms/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiCharge.root",
                        help="Input LSTNtuple ROOT file")
    parser.add_argument("--pdf", type=str, default="phis.pdf",
                        help="Path to save the output plots in PDF format")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode for additional output")
    parser.add_argument("--events", type=int, default=1000,
                        help="Maximum number of events to process")
    return parser.parse_args()


def main() -> None:
    args = options()
    plotter = PhiPlotter(args.events, args.debug)
    plotter.load_data(args.ntuple)
    plotter.make_track_pair_dataframe()


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
        dfs = []
        for i_ev in range(self.n_ev):
            if i_ev >= self.max_ev:
                break
            dfs.append( self.make_track_pair_dataframe_one_event(i_ev) )
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Total track pairs found: {len(self.df)}")
        if self.debug:
            print("First 10 pairs:\n", self.df.head(10))


    def make_track_pair_dataframe_one_event(self, i_ev: int) -> pd.DataFrame:
        t5_idx, pls_idx = self.match_tracks_one_event(i_ev)
        return pd.DataFrame({
            "event": i_ev,
            "t5_pt": self.data["t5_pt"][i_ev][t5_idx],
            "t5_eta": self.data["t5_eta"][i_ev][t5_idx],
            "t5_phi": self.data["t5_phi"][i_ev][t5_idx],
            "pLS_ptIn": self.data["pLS_ptIn"][i_ev][pls_idx],
            "pLS_eta": self.data["pLS_eta"][i_ev][pls_idx],
            "pLS_phi": self.data["pLS_phi"][i_ev][pls_idx],
            "pLS_deltaPhi": self.data["pLS_deltaPhi"][i_ev][pls_idx],
            "t5_simIdx": self.data["t5_simIdx"][i_ev][t5_idx],
            "pLS_simIdx": self.data["pLS_simIdx"][i_ev][pls_idx],
            "t5_idx": t5_idx,
            "pLS_idx": pls_idx
        })


    def match_t5_pls_tracks(self) -> None:
        pairs = []
        for i_ev in range(self.n_ev):
            if i_ev >= self.max_ev:
                break
            pairs.append( self.match_tracks_one_event(i_ev) )
        self.pairs = np.concatenate(pairs)
        print(f"Total matching pairs found: {len(self.pairs)}")
        if self.debug:
            print("First 10 pairs:\n", self.pairs[:10])


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




if __name__ == "__main__":
    main()
