"""
A script to get sim features from a trackingNtuple which are matched to features produced for embedding training

NB: This assumes events are ordered in the tracking ntuple and LSTNtuple the same way!
Be careful when producing LSTNtuples with multiple streams. (Easy approach: process with 1 stream.)
"""

import argparse
import awkward as ak
import numpy as np
import pickle
import time
import uproot
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from matplotlib import rcParams
rcParams.update({"font.size": 18})

k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2

TRACKING_NTUPLE = "/ceph/cms/store/user/evourlio/LST/samples/CMSSW_12_2_0_pre2/RelValTTbar_14TeV_CMSSW_12_5_0_pre3/event_1000.root"

BRANCHES = [
    "sim_eta",
    "sim_pca_dxy",
    "sim_pca_dz",
    "sim_pdgId",
    "sim_phi",
    "sim_pt",
    "sim_q",
    # "sim_vx",
    # "sim_vy",
    # "sim_vz",
]


def main():
    args = options()
    writer = SimFeatureWriter(features_t5=args.features_t5,
                              features_pls=args.features_pls,
                              tracking_ntuple=args.tracking_ntuple,
                              output_t5=args.output_t5,
                              output_pls=args.output_pls,
                              pdf=args.pdf,
                              )
    writer.get_truth_features()
    writer.plot()
    writer.write()


def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--features_t5", type=str, default="features_t5.pkl",
                        help="Path to the precomputed T5 features file")
    parser.add_argument("--features_pls", type=str, default="features_pls.pkl",
                        help="Path to the precomputed PLS features file")
    parser.add_argument("--tracking_ntuple", type=str, default=TRACKING_NTUPLE,
                        help="Path to the tracking ntuple file")
    parser.add_argument("--output_t5", type=str, default="sim_features_t5.pkl",
                        help="Path to save the output T5 features")
    parser.add_argument("--output_pls", type=str, default="sim_features_pls.pkl",
                        help="Path to save the output PLS features")
    parser.add_argument("--pdf", type=str, default="sim_features.pdf",
                        help="Path to save the output plots in PDF format")
    return parser.parse_args()


class SimFeatureWriter:


    def __init__(self,
                 features_t5,
                 features_pls,
                 tracking_ntuple,
                 output_t5,
                 output_pls,
                 pdf,
                 ):

        (self.features_per_event,
         self.displaced_per_event,
         self.sim_indices_per_event) = self.load_t5_features(features_t5)

        (self.pls_features_per_event,
         self.pls_sim_indices_per_event) = self.load_pls_features(features_pls)

        self.sim = self.load_tracking_ntuple(tracking_ntuple, BRANCHES)

        self.pdf = pdf
        self.output_t5 = output_t5
        self.output_pls = output_pls


    def load_t5_features(self, features_t5):
        print(f"Loading T5 features from {features_t5} ...")
        with open(features_t5, "rb") as fi:
            data = pickle.load(fi)
        return [
            data["features_per_event"],
            data["displaced_per_event"],
            data["sim_indices_per_event"],
        ]


    def load_pls_features(self, features_pls):
        print(f"Loading PLS features from {features_pls} ...")
        with open(features_pls, "rb") as fi:
            data = pickle.load(fi)
        return [
            data["pLS_features_per_event"],
            data["pLS_sim_indices_per_event"],
        ]


    def load_tracking_ntuple(self, file_path, branches):
        parquet_file = Path("tracking_ntuple_sim.parquet")
        if parquet_file.exists():
            print(f"Loading tracking ntuple from {parquet_file} ...")
            return ak.from_parquet(parquet_file)
        else:
            print(f"Loading tracking ntuple from {file_path} ...")
            with uproot.open(f"{file_path}:trackingNtuple/tree") as tr:
                print(tr)
                data = tr.arrays(branches)
                print(f"Writing tracking ntuple to {parquet_file} ...")
                ak.to_parquet(data, parquet_file)
                return data


    def get_truth_features(self):
        self.get_truth_features_t5()
        self.get_truth_features_pls()


    def get_truth_features_t5(self):
        print("Getting T5 sim features ... ")
        n_ev = len(self.sim_indices_per_event)
        sim_features_t5 = {br: [] for br in BRANCHES}
        for ev in tqdm(range(n_ev)):
            idxs = self.sim_indices_per_event[ev]
            for br in BRANCHES:
                sim_features_t5[br].append(self.sim[br][ev][idxs].to_numpy())
        self.sim_features_t5 = sim_features_t5


    def get_truth_features_pls(self):
        print("Getting PLS sim features ... ")
        n_ev = len(self.pls_sim_indices_per_event)
        sim_features_pls = {br: [] for br in BRANCHES}
        for ev in tqdm(range(n_ev)):
            idxs = self.pls_sim_indices_per_event[ev]
            for br in BRANCHES:
                sim_features_pls[br].append(self.sim[br][ev][idxs].to_numpy())
        self.sim_features_pls = sim_features_pls


    def plot(self):
        with PdfPages(self.pdf) as pdf:
            self.plot_diffs(pdf)
            self.plot_features(pdf)


    def plot_diffs(self, pdf: PdfPages):

        def normalize_angle(arr):
            arr[arr > np.pi] -= 2 * np.pi
            arr[arr < -np.pi] += 2 * np.pi
            # return arr

        n_ev = len(self.features_per_event)
        cat = np.concatenate
        diff = {}

        # diffs of physics features
        diff["Sim 1/pT - T5 1/pT"] = 1.0/cat(self.sim_features_t5["sim_pt"]) - cat([self.features_per_event[ev][:, 21] / (k2Rinv1GeVf * 2) for ev in range(n_ev)])
        diff["Sim eta - T5 eta"] = cat(self.sim_features_t5["sim_eta"]) - cat([self.features_per_event[ev][:, 0] for ev in range(n_ev)]) * 2.5
        diff["Sim phi - T5 phi"] = cat(self.sim_features_t5["sim_phi"]) - cat([np.atan2(self.features_per_event[ev][:, 2],
                                                                                        self.features_per_event[ev][:, 1]) for ev in range(n_ev)])

        diff["Sim 1/pT - pLS 1/pT"] = 1.0/cat(self.sim_features_pls["sim_pt"]) - cat([np.abs(self.pls_features_per_event[ev][:, 4]) for ev in range(n_ev)])
        diff["Sim eta - pLS eta"] = cat(self.sim_features_pls["sim_eta"]) - cat([self.pls_features_per_event[ev][:, 0] for ev in range(n_ev)]) * 4.0
        diff["Sim phi - pLS phi"] = cat(self.sim_features_pls["sim_phi"]) - cat([np.atan2(self.pls_features_per_event[ev][:, 3],
                                                                                          self.pls_features_per_event[ev][:, 2]) for ev in range(n_ev)])

        # normalize dphis
        normalize_angle(diff["Sim phi - T5 phi"])
        normalize_angle(diff["Sim phi - pLS phi"])

        # define binnings
        bins = {
            "Sim 1/pT - T5 1/pT": np.linspace(-0.4, 0.4, 100),
            "Sim eta - T5 eta": np.linspace(-0.4, 0.4, 100),
            "Sim phi - T5 phi": np.linspace(-0.4, 0.4, 100),
            "Sim 1/pT - pLS 1/pT": np.linspace(-0.4, 0.4, 100),
            "Sim eta - pLS eta": np.linspace(-0.04, 0.04, 100),
            "Sim phi - pLS phi": np.linspace(-0.4, 0.4, 100),
        }

        # diffs of a few selected features
        args = dict(color="green",
                    edgecolor="black",
                    histtype="stepfilled")
        for name in diff:
            track_type = "T5" if "T5" in name else "pLS"
            fig, ax = plt.subplots(figsize=(8, 8))
            std = np.std(diff[name])
            p68 = np.percentile(np.abs(diff[name]), 68)
            p95 = np.percentile(np.abs(diff[name]), 95)
            args["color"] = "skyblue" if track_type == "T5" else "pink"
            ax.hist(diff[name], bins=bins[name], **args)
            ax.set_xlabel(name)
            ax.set_ylabel("Tracks")
            ax.text(0.05, 1.01, f"Sim. vs. {track_type}", transform=ax.transAxes)
            # ax.text(0.73, 1.03, f"Std = {std:.4f}", transform=ax.transAxes)
            ax.text(0.42, 1.01, f"68% @ {p68:.3f}, 95% @ {p95:.3f}", transform=ax.transAxes)
            ax.grid()
            ax.set_axisbelow(True)
            ax.tick_params(right=True, top=True, direction="in")
            fig.subplots_adjust(left=0.17, right=0.96, top=0.95, bottom=0.1)
            pdf.savefig(fig)
            plt.close(fig)


    def plot_features(self, pdf: PdfPages):

        # bare sim features
        args = dict(edgecolor="black",
                    histtype="stepfilled")
        for coll, sim_features in [("T5", self.sim_features_t5),
                                   ("pLS", self.sim_features_pls),
                                   ]:
            print(f"Plotting {coll} sim features ...")
            color = "blue" if coll == "T5" else "red"
            for br in BRANCHES:
                bins = 100
                if "sim_pca_dxy" in br:
                    xmax = 0.005 if coll == "T5" else 0.005
                    # xmax = 0.5 if coll == "T5" else 0.5
                    bins = np.linspace(-xmax, xmax, 101)
                elif "sim_pca_dz" in br:
                    xmax = 20 if coll == "T5" else 20
                    bins = np.linspace(-xmax, xmax, 101)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.hist(np.concatenate(sim_features[br]), bins=bins, color=color, **args)
                ax.set_title(f"Matched to {coll} tracks")
                ax.set_xlabel(br)
                ax.set_ylabel("Tracks")
                ax.grid()
                ax.set_axisbelow(True)
                fig.subplots_adjust(left=0.17, right=0.96, top=0.95, bottom=0.1)
                pdf.savefig(fig)
                if "sim_pca_" in br:
                    ax.semilogy()
                    pdf.savefig(fig)
                plt.close(fig)

            # bonus plots: q/pt
            bins = np.linspace(-2, 2, 101)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(np.concatenate(sim_features["sim_q"]) / np.concatenate(sim_features["sim_pt"]),
                    bins=bins, color=color, **args)
            ax.set_title(f"Matched to {coll} tracks")
            ax.set_xlabel("sim_q / sim_pt")
            ax.set_ylabel("Tracks")
            ax.grid()
            ax.set_axisbelow(True)
            fig.subplots_adjust(left=0.17, right=0.96, top=0.95, bottom=0.1)
            pdf.savefig(fig)
            plt.close(fig)


    def write(self):
        print(f"Writing T5 sim features to {self.output_t5} ...")
        with open(self.output_t5, "wb") as fo:
            pickle.dump({"sim_features": self.sim_features_t5}, fo)

        print(f"Writing pLS sim features to {self.output_pls} ...")
        with open(self.output_pls, "wb") as fo:
            pickle.dump({"sim_features": self.sim_features_pls}, fo)


if __name__ == "__main__":
    main()
