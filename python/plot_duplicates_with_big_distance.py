import awkward as ak
import numpy as np
from pathlib import Path
import pickle
import uproot
from tqdm import tqdm

# from preprocess import load_root_file
from preprocess import branches_list
from ml import EmbeddingNetT5, EmbeddingNetpLS

BRANCHES = Path("branches.pkl")

def main():
    network = Path("model_weights.pth")
    file_path = Path("/ceph/users/atuna/work/gavin_tc_dnn/python/pls_t5_embed.root")

    # print(f"Loading ak branches from {file_path} ...")
    # data = load_root_file_ak(file_path, branches_list)
    # print(f"Loaded")
    # return

    plotter = PlotDuplicatesWithBigDistance()
    plotter.load_raw_data(file_path)
    plotter.get_t5_features()
    plotter.get_t5_t5_pairs()
    # plotter.get_t5_pls_pairs()
    pairs = plotter.get_pls_t5_pairs_of_interest()
    plotter.plot(pairs)


class PlotDuplicatesWithBigDistance:
    def __init__(self):
        # if BRANCHES.exists():
        #     print(f"Loading branches from {BRANCHES} ...")
        #     with open(BRANCHES, "rb") as fi:
        #         self.branches = pickle.load(fi)
        # else:
        #     print(f"Loading branches from {root_path} ...")
        #     self.branches = load_root_file(root_path)
        #     with open(BRANCHES, "wb") as fi:
        #         pickle.dump(self.branches, fi)
        pass


    def load_raw_data(self, file_path):
        self.data = load_root_file_ak(file_path, branches_list)


    def get_t5_features(self):
        pass


    def get_t5_t5_pairs(self):
        pass


    def get_pls_t5_pairs_of_interest(self):
        return [
            (199, 2_588, 536),
        ]


    def plot(self, pairs):
        pass


def load_root_file_ak(file_path, branches):
    with uproot.open(f"{file_path}:tree") as tr:
        print(tr)
        return tr.arrays(branches)


def load_t5_features():
    with open(FEATURES_T5, "rb") as fi:
        data = pickle.load(fi)
    return [
        data["features_per_event"],
        data["displaced_per_event"],
        data["sim_indices_per_event"],
    ]


def load_pls_features():
    with open(FEATURES_PLS, "rb") as fi:
        data = pickle.load(fi)
    return [
        data["pLS_features_per_event"],
        data["pLS_sim_indices_per_event"],
    ]

if __name__ == "__main__":
    main()
