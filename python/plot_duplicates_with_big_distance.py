import awkward as ak
import numpy as np
from pathlib import Path
import pickle
import uproot
import torch

# from preprocess import load_root_file
from preprocess import branches_list
from ml import EmbeddingNetT5, EmbeddingNetpLS

PAIRS_T5T5 = Path("pairs_t5t5.pkl")
PAIRS_T5PLS = Path("pairs_t5pls.pkl")
BRANCHES = Path("branches.pkl")
BONUS_FEATURES = 2
SIM_BRANCHES = [
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_pdgId",
]
TRACKING_NTUPLE = Path("/ceph/cms/store/user/evourlio/LST/samples/CMSSW_12_2_0_pre2/RelValTTbar_14TeV_CMSSW_12_5_0_pre3/event_1000.root")


def main():
    network = Path("model_weights.pth")
    file_path = Path("/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75.root")
    # file_path = Path("/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p50.root")

    plotter = PlotDuplicatesWithBigDistance()
    plotter.load_network(network)
    plotter.load_t5_t5_pairs()
    plotter.load_t5_pls_pairs()
    plotter.load_test_scores()
    plotter.load_tracking_ntuple(TRACKING_NTUPLE, SIM_BRANCHES)
    plotter.load_raw_data(file_path)
    pairs = plotter.get_pls_t5_pairs_of_interest()
    plotter.plot_t5_pls_pairs(pairs)

    pairs_t5t5 = plotter.get_t5_t5_pairs_of_interest()
    plotter.plot_t5_t5_pairs(pairs_t5t5)


class PlotDuplicatesWithBigDistance:
    def __init__(self):
        self.embed_t5 = EmbeddingNetT5()
        self.embed_pls = EmbeddingNetpLS()


    def load_network(self, path: Path):
        print(f"Loading model from {path} ...")
        checkpoint = torch.load(path)
        self.embed_t5.load_state_dict(checkpoint["embed_t5"])
        self.embed_pls.load_state_dict(checkpoint["embed_pls"])
        self.embed_t5.eval()
        self.embed_pls.eval()


    def load_test_scores(self):
        self.load_t5_t5_test_scores()
        self.load_pls_t5_test_scores()


    def load_t5_t5_test_scores(self):
        print("Calculating T5-T5 scores ...")
        with torch.no_grad():
            x_left = torch.tensor(self.X_left_test[:, :-BONUS_FEATURES])
            x_right = torch.tensor(self.X_right_test[:, :-BONUS_FEATURES])
            e_l = self.embed_t5(x_left)
            e_r = self.embed_t5(x_right)
            self.d_t5t5 = torch.sqrt(((e_l - e_r) ** 2).sum(dim=1, keepdim=True) + 1e-6)
        self.d_t5t5 = self.d_t5t5.numpy().flatten()


    def load_pls_t5_test_scores(self):
        print("Calculating PLS-T5 scores ...")
        with torch.no_grad():
            x_pls = torch.tensor(self.X_pls_test[:, :-BONUS_FEATURES])
            x_t5 = torch.tensor(self.X_t5raw_test[:, :-BONUS_FEATURES])
            e_pls = self.embed_pls(x_pls)
            e_t5 = self.embed_t5(x_t5)
            self.d_plst5 = torch.sqrt(((e_pls - e_t5) ** 2).sum(dim=1, keepdim=True) + 1e-6)
        self.d_plst5 = self.d_plst5.numpy().flatten()


    def load_raw_data(self, file_path):
        print(f"Loading raw data from {file_path} ...")
        self.data = load_root_file_ak(file_path, branches_list + SIM_BRANCHES)


    def load_tracking_ntuple(self, file_path, branches):
        print(f"Loading tracking ntuple from {file_path} ...")
        self.sim = load_tracking_ntuple(file_path, branches)


    def load_t5_t5_pairs(self):
        print("Getting T5-T5 pairs")
        [self.X_left_train,
         self.X_left_test,
         self.X_right_train,
         self.X_right_test,
         self.y_t5_train,
         self.y_t5_test,
         self.w_t5_train,
         self.w_t5_test,
         self.true_L_train,
         self.true_L_test,
         self.true_R_train,
         self.true_R_test
         ] = load_t5_t5_pairs()


    def load_t5_pls_pairs(self):
        print("Getting PLS-T5 pairs")
        [self.X_pls_train,
         self.X_pls_test,
         self.X_t5raw_train,
         self.X_t5raw_test,
         self.y_pls_train,
         self.y_pls_test,
         self.w_pls_train,
         self.w_pls_test,
         ] = load_t5_pls_pairs()


    def get_t5_features(self):
        pass


    def get_t5_t5_pairs_of_interest(self, max_pairs=2):
        dup_mask = self.y_t5_test == 0
        dup_dist = self.d_t5t5[dup_mask]
        dup_x_l = self.X_left_test[dup_mask]
        dup_x_r = self.X_right_test[dup_mask]
        dup_idxs = np.flip(np.argsort(dup_dist))
        ret = []
        for i in range(max_pairs):
            idx = dup_idxs[i]
            ev_l, i_l = dup_x_l[idx][-2], dup_x_l[idx][-1]
            ev_r, i_r = dup_x_r[idx][-2], dup_x_r[idx][-1]
            if ev_l != ev_r:
                raise ValueError(f"Event mismatch: {ev_l} != {ev_r}")
            ret.append( (int(ev_l), int(i_l), int(i_r), dup_dist[idx]) )
        return ret


    def get_pls_t5_pairs_of_interest(self, max_pairs=5):
        # Get pairs of PLS-T5 duplicates with the largest distance
        dup_mask = self.y_pls_test == 0
        dup_dist = self.d_plst5[dup_mask]
        dup_x_t5 = self.X_t5raw_test[dup_mask]
        dup_x_pls = self.X_pls_test[dup_mask]
        dup_idxs = np.flip(np.argsort(dup_dist))
        ret = []
        for i in range(max_pairs):
            idx = dup_idxs[i]
            ev_pls, i_pls = dup_x_pls[idx][-2], dup_x_pls[idx][-1]
            ev_t5, i_t5   = dup_x_t5[idx][-2], dup_x_t5[idx][-1]
            if ev_pls != ev_t5:
                raise ValueError(f"Event mismatch: {ev_pls} != {ev_t5}")
            ret.append( (int(ev_pls), int(i_pls), int(i_t5), dup_dist[idx]) )
        return ret
        # return [
        #     (199, 2_588, 536),
        # ]


    def plot_t5_t5_pairs(self, pairs):
        pass


    def plot_t5_pls_pairs(self, pairs):
        for (ev, i_pls, i_t5, dist) in pairs:
            pls_eta, pls_phi = self.data["pLS_eta"][ev][i_pls], self.data["pLS_phi"][ev][i_pls]
            t5_eta, t5_phi = self.data["t5_eta"][ev][i_t5], self.data["t5_phi"][ev][i_t5]
            pls_simidx = self.data["pLS_matched_simIdx"][ev][i_pls]
            t5_simidx = self.data["t5_matched_simIdx"][ev][i_t5]
            t5_pmatched = self.data["t5_pMatched"][ev][i_t5]
            deltar2 = (pls_eta - t5_eta) ** 2 + (pls_phi - t5_phi) ** 2
            if pls_simidx != t5_simidx or len(pls_simidx) != 1:
                print("WARNING: simidx are weird!")
            simidx = pls_simidx[0]
            sim_pt, sim_eta, sim_phi, sim_pdgId = (
                self.sim["sim_pt"][ev][simidx],
                self.sim["sim_eta"][ev][simidx],
                self.sim["sim_phi"][ev][simidx],
                self.sim["sim_pdgId"][ev][simidx],
            )
            print("")
            print(f"Event: {ev}")
            print(f"pLS index: {i_pls}, T5 index: {i_t5}")
            print(pls_eta, pls_phi)
            print(t5_eta, t5_phi)
            print(f"pLS matched simIdx: {pls_simidx}")
            print(f"T5 matched simIdx: {t5_simidx}")
            print(f"T5 pMatched: {t5_pmatched}")
            print(f"Delta R2: {deltar2:.3f}")
            print(f"Delta R: {np.sqrt(deltar2):.3f}")
            print(f"Distance: {dist:.3f}")
            print(f"Sim index: {simidx} vs len(sim_pt): {len(self.sim['sim_pt'][ev])}")
            print(f"Sim: {sim_pdgId} pt={sim_pt:6.2f}, eta={sim_eta:6.3f}, phi={sim_phi:6.3f}")
            print("")
            # for xev in range(1000):
            #     sim_pt, sim_eta, sim_phi, sim_pdgId = (
            #         self.sim["sim_pt"][xev][simidx],
            #         self.sim["sim_eta"][xev][simidx],
            #         self.sim["sim_phi"][xev][simidx],
            #         self.sim["sim_pdgId"][xev][simidx],
            #     )
            #     dr = np.sqrt((pls_eta - sim_eta) ** 2 + (pls_phi - sim_phi) ** 2)
            #     blob = " <-- " if dr < 0.2 else ""
            #     print(f"  {xev} Sim: {sim_pdgId:5} pt={sim_pt:6.2f}, eta={sim_eta:6.3f}, phi={sim_phi:6.3f}  --  DR: {dr:.3f} {blob}")



def load_root_file_ak(file_path, branches):
    # with uproot.open(f"{file_path}") as fi:
    #     print(fi)
    #     for key in fi.keys():
    #         print(f"File key: {key}")
    with uproot.open(f"{file_path}:tree") as tr:
        print(tr)
        # for key in tr.keys():
        #     print(f"Key: {key}")
        return tr.arrays(branches)


def load_tracking_ntuple(file_path, branches):
    parquet_file = Path("tracking_ntuple.parquet")
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


def load_t5_t5_pairs():
    with open(PAIRS_T5T5, "rb") as fi:
        data = pickle.load(fi)
    return [
        data["X_left_train"],
        data["X_left_test"],
        data["X_right_train"],
        data["X_right_test"],
        data["y_t5_train"],
        data["y_t5_test"],
        data["w_t5_train"],
        data["w_t5_test"],
        data["true_L_train"],
        data["true_L_test"],
        data["true_R_train"],
        data["true_R_test"]
    ]


def load_t5_pls_pairs():
    with open(PAIRS_T5PLS, "rb") as fi:
        data = pickle.load(fi)
    return [
        data["X_pls_train"],
        data["X_pls_test"],
        data["X_t5raw_train"],
        data["X_t5raw_test"],
        data["y_pls_train"],
        data["y_pls_test"],
        data["w_pls_train"],
        data["w_pls_test"],
    ]


if __name__ == "__main__":
    main()
