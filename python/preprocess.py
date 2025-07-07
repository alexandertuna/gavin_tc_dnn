import os
import pickle
import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import awkward as ak # Using awkward array for easier handling of jagged data
import time # For timing steps
from tqdm import tqdm

import time, random, math, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

DELTA_R2_CUT = 0.02
ETA_MAX = 2.5

# pairing hyper-parameters
DELTA_R2_CUT_PLS_T5 = 0.02
DISP_VXY_CUT       = 0.1
INVALID_SIM_IDX    = -1
MAX_SIM            = 1000
MAX_DIS            = 1000

# intermediate data
LOAD_FEATURES = False
FEATURES_T5 = Path("features_t5.pkl")
FEATURES_PLS = Path("features_pls.pkl")
LOAD_PAIRS = False
PAIRS_T5T5 = Path("pairs_t5t5.pkl")
PAIRS_T5PLS = Path("pairs_t5pls.pkl")


def load_root_file(file_path, branches=None, print_branches=False):
    all_branches = {}
    print(f"Loading ROOT file: {file_path}")
    with uproot.open(file_path) as file:
        tree = file["tree"]
        # Load all ROOT branches into array if not specified
        if branches is None:
            branches = tree.keys()
        # Option to print the branch names
        if print_branches:
            print("Branches:", tree.keys())
        # Each branch is added to the dictionary
        for branch in tqdm(branches):
            try:
                all_branches[branch] = (tree[branch].array(library="np"))
            except uproot.KeyInFileError as e:
                print(f"KeyInFileError: {e}")
        # Number of events in file
        all_branches['event'] = tree.num_entries
    return all_branches

branches_list = [
    't5_innerRadius',
    't5_bridgeRadius',
    't5_outerRadius',
    't5_pt',
    't5_eta',
    't5_phi',
    't5_isFake',
    't5_t3_idx0',
    't5_t3_idx1',

    't5_t3_fakeScore1',
    't5_t3_promptScore1',
    't5_t3_displacedScore1',
    't5_t3_fakeScore2',
    't5_t3_promptScore2',
    't5_t3_displacedScore2',

    't5_pMatched',
    't5_sim_vxy',
    't5_sim_vz',
    't5_matched_simIdx'
]

branches_list += [
    'pLS_eta',
    'pLS_etaErr',
    'pLS_phi',
    'pLS_matched_simIdx',
    'pLS_circleCenterX',
    'pLS_circleCenterY',
    'pLS_circleRadius',
    'pLS_ptIn',
    'pLS_ptErr',
    'pLS_px',
    'pLS_py',
    'pLS_pz',
    'pLS_isQuad',
    'pLS_isFake'
]

# Hit-dependent branches
suffixes = ['r', 'z', 'eta', 'phi', 'layer']
branches_list += [f't5_t3_{i}_{suffix}' for i in [0, 2, 4] for suffix in suffixes]

def delta_phi(phi1, phi2):
    delta = phi1 - phi2
    # Adjust delta to be within the range [-pi, pi]
    if delta > np.pi:
        delta -= 2 * np.pi
    elif delta < -np.pi:
        delta += 2 * np.pi
    return delta

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

class Preprocessor:

    def __init__(self, root_path):

        self.root_path = root_path
        branches = self.load_root_file(root_path) if not LOAD_FEATURES else None

        print("Getting T5 features")
        [features_per_event,
         displaced_per_event,
         sim_indices_per_event] = self.get_t5_features(branches) if not LOAD_FEATURES else load_t5_features()
        self.features_per_event = features_per_event
        self.sim_indices_per_event = sim_indices_per_event

        print("Getting PLS features")
        [pLS_features_per_event,
         pLS_sim_indices_per_event] = self.get_pls_features(branches) if not LOAD_FEATURES else load_pls_features()

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
         ] = self.get_t5_pairs(features_per_event,
                               displaced_per_event,
                               sim_indices_per_event) if not LOAD_PAIRS else load_t5_t5_pairs()

        print("Getting PLS-T5 pairs")
        [self.X_pls_train,
         self.X_pls_test,
         self.X_t5raw_train,
         self.X_t5raw_test,
         self.y_pls_train,
         self.y_pls_test,
         self.w_pls_train,
         self.w_pls_test,
         ] = self.get_t5_pls_pairs(pLS_features_per_event,
                                   pLS_sim_indices_per_event,
                                   features_per_event,
                                   displaced_per_event,
                                   sim_indices_per_event) if not LOAD_PAIRS else load_t5_pls_pairs()


    def load_root_file(self, root_path):

        print("Loading ROOT file:", root_path)
        return load_root_file(root_path, branches_list, print_branches=True)

    def speed_test(self):

        if not LOAD_FEATURES:
            raise ValueError("LOAD_FEATURES must be True to run the speed test")

        # load T5 and pLS features
        [features_per_event,
         displaced_per_event,
         sim_indices_per_event] = load_t5_features()

        [pLS_features_per_event,
         pLS_sim_indices_per_event] = load_pls_features()

        evts = range(500)
        max_per_event = 1e6 # 1000 # 1e6
        invalid_sim_idx = -1

        # test T5-T5 pairs
        for evt_idx in evts:
            work_args = (evt_idx,
                        features_per_event[evt_idx],
                        sim_indices_per_event[evt_idx],
                        displaced_per_event[evt_idx],
                        max_per_event,
                        max_per_event,
                        invalid_sim_idx,
                        )

            # run both versions
            random.seed(42)
            evt_idx, sim_pairs, dis_pairs = _pairs_single_event(*work_args)
            random.seed(42)
            evt_idx, sim_pairs_vec, dis_pairs_vec = _pairs_single_event_vectorized(*work_args)

            # compare
            sim_pairs_vec = [(i, j) for i, j in sim_pairs_vec] # de-numpify
            dis_pairs_vec = [(i, j) for i, j in dis_pairs_vec]
            sim_equal = (sim_pairs == sim_pairs_vec)
            dis_equal = (dis_pairs == dis_pairs_vec)
            sim_equiv = sorted(sim_pairs) == sorted(sim_pairs_vec)
            dis_equiv = sorted(dis_pairs) == sorted(dis_pairs_vec)
            print(f"[evt {evt_idx:4d}]  sim. pairs equal: {sim_equal}. Equivalent: {sim_equiv}")
            print(f"[evt {evt_idx:4d}]  dis. pairs equal: {dis_equal}. Equivalent: {dis_equiv}")
            print("")

        # test pLS-T5 pairs
        for evt_idx in evts:
            works_args = (
                evt_idx,
                pLS_features_per_event[evt_idx],
                pLS_sim_indices_per_event[evt_idx],
                features_per_event[evt_idx],
                sim_indices_per_event[evt_idx],
                displaced_per_event[evt_idx],
                max_per_event,
                max_per_event,
                invalid_sim_idx,
            )

            # run both versions
            random.seed(42)
            _, packed = _pairs_pLS_T5_single(*works_args)
            random.seed(42)
            _, packed_vec = _pairs_pLS_T5_single_vectorized(*works_args)

            # compare
            def tup_equal(a, b):
                return (np.array_equal(a[0], b[0]) and
                        np.array_equal(a[1], b[1]) and
                        a[2] == b[2] and
                        np.array_equal(a[3], b[3]))
            
            def packed_equal(a, b):
                if len(a) != len(b):
                    return False
                for (a_i, b_i) in zip(a, b):
                    if not tup_equal(a_i, b_i):
                        return False
                return True

            equal = packed_equal(packed, packed_vec)
            print(f"[evt {evt_idx:4d}]  pLS-T5 pairs equal: {equal}")
            print("")


    def parallelism_test(self):

        if not LOAD_FEATURES:
            raise ValueError("LOAD_FEATURES must be True to run the speed test")

        # load T5 and pLS features
        [features_per_event,
         displaced_per_event,
         sim_indices_per_event] = load_t5_features()

        # test parallelism with different configurations
        results = {}
        for vectorize in [True, False]:
            for n_workers in [256, 192, 128, 96, 64, 32, 16, 12, 8, 6, 4]:
                for it in range(3):
                    print(f"Testing with vectorize={vectorize}, n_workers={n_workers}, it={it}")
                    start = time.time()
                    _ = create_t5_pairs_balanced_parallel(
                        features_per_event,
                        sim_indices_per_event,
                        displaced_per_event,
                        max_similar_pairs_per_event    = 1000,
                        max_dissimilar_pairs_per_event = 1000,
                        invalid_sim_idx                = -1,
                        n_workers                      = n_workers,
                        vectorize                      = vectorize,
                    )
                    end = time.time()
                    results[(vectorize, n_workers, it)] = end - start
                    print(f"Result vectorize={vectorize} n_workers={n_workers} it={it} {end - start:.4f} seconds")


        # print results
        print("Parallelism test results:")
        for (vectorize, n_workers, it), duration in results.items():
            print(f"vectorize={vectorize}, n_workers={n_workers}, it={it}: {duration:.4f} seconds")


    def get_t5_features(self, branches):

        z_max = np.max([np.max(event) for event in branches[f't5_t3_4_z']])
        r_max = np.max([np.max(event) for event in branches[f't5_t3_4_r']])
        # eta_max = 2.5
        phi_max = np.pi
        n_events = np.shape(branches['t5_pt'])[0]
        print(f'Z max: {z_max}, R max: {r_max}, Eta max: {ETA_MAX}')

        # ------------------------------------------------------------------

        pMATCHED_THRESHOLD = 0.        # keep if t5_pMatched ≥ this
        print(f"\nBuilding T5 features  (pMatched ≥ {pMATCHED_THRESHOLD}) …")

        features_per_event    = []
        eta_per_event         = []
        displaced_per_event = []
        sim_indices_per_event = []

        kept_tot, init_tot = 0, 0
        for ev in tqdm(range(n_events)):

            n_t5 = len(branches['t5_t3_idx0'][ev])
            init_tot += n_t5
            if n_t5 == 0:
                continue

            feat_evt = []
            eta_evt  = []
            sim_evt  = []
            disp_evt = []

            for i in range(n_t5):
                if branches['t5_pMatched'][ev][i] < pMATCHED_THRESHOLD:
                    continue

                idx0 = branches['t5_t3_idx0'][ev][i]
                idx1 = branches['t5_t3_idx1'][ev][i]

                # hit-level quantities -------------------------------------------------
                eta1 = (branches['t5_t3_0_eta'][ev][idx0])
                eta2 = abs(branches['t5_t3_2_eta'][ev][idx0])
                eta3 = abs(branches['t5_t3_4_eta'][ev][idx0])
                eta4 = abs(branches['t5_t3_2_eta'][ev][idx1])
                eta5 = abs(branches['t5_t3_4_eta'][ev][idx1])

                phi1 = branches['t5_t3_0_phi'][ev][idx0]
                phi2 = branches['t5_t3_2_phi'][ev][idx0]
                phi3 = branches['t5_t3_4_phi'][ev][idx0]
                phi4 = branches['t5_t3_2_phi'][ev][idx1]
                phi5 = branches['t5_t3_4_phi'][ev][idx1]

                z1 = abs(branches['t5_t3_0_z'][ev][idx0])
                z2 = abs(branches['t5_t3_2_z'][ev][idx0])
                z3 = abs(branches['t5_t3_4_z'][ev][idx0])
                z4 = abs(branches['t5_t3_2_z'][ev][idx1])
                z5 = abs(branches['t5_t3_4_z'][ev][idx1])

                r1 = branches['t5_t3_0_r'][ev][idx0]
                r2 = branches['t5_t3_2_r'][ev][idx0]
                r3 = branches['t5_t3_4_r'][ev][idx0]
                r4 = branches['t5_t3_2_r'][ev][idx1]
                r5 = branches['t5_t3_4_r'][ev][idx1]

                inR  = branches['t5_innerRadius' ][ev][i]
                brR  = branches['t5_bridgeRadius'][ev][i]
                outR = branches['t5_outerRadius' ][ev][i]

                s1_fake   = branches['t5_t3_fakeScore1'     ][ev][i]
                s1_prompt = branches['t5_t3_promptScore1'   ][ev][i]
                s1_disp   = branches['t5_t3_displacedScore1'][ev][i]
                d_fake    = branches['t5_t3_fakeScore2'     ][ev][i] - s1_fake
                d_prompt  = branches['t5_t3_promptScore2'   ][ev][i] - s1_prompt
                d_disp    = branches['t5_t3_displacedScore2'][ev][i] - s1_disp

                f = [
                    eta1 / ETA_MAX,
                    np.cos(phi1),
                    np.sin(phi1),
                    z1 / z_max,
                    r1 / r_max,

                    eta2 - abs(eta1),
                    delta_phi(phi2, phi1),
                    (z2 - z1) / z_max,
                    (r2 - r1) / r_max,

                    eta3 - eta2,
                    delta_phi(phi3, phi2),
                    (z3 - z2) / z_max,
                    (r3 - r2) / r_max,

                    eta4 - eta3,
                    delta_phi(phi4, phi3),
                    (z4 - z3) / z_max,
                    (r4 - r3) / r_max,

                    eta5 - eta4,
                    delta_phi(phi5, phi4),
                    (z5 - z4) / z_max,
                    (r5 - r4) / r_max,

                    1.0 / inR,
                    1.0 / brR,
                    1.0 / outR,

                    s1_fake, s1_prompt, s1_disp,
                    d_fake,  d_prompt,  d_disp
                ]
                feat_evt.append(f)
                eta_evt.append(eta1)
                disp_evt.append(branches['t5_sim_vxy'][ev][i])

                # first (or only) matched sim-index, -1 if none -----------------------
                simIdx_list = branches['t5_matched_simIdx'][ev][i]
                sim_evt.append(simIdx_list[0] if len(simIdx_list) else -1)

            # push to global containers ----------------------------------------------
            if feat_evt:                                # skip events with no survivors
                features_per_event.append(np.asarray(feat_evt, dtype=np.float32))
                eta_per_event.append(np.asarray(eta_evt,  dtype=np.float32))
                displaced_per_event.append(np.asarray(disp_evt, dtype=np.float32))
                sim_indices_per_event.append(np.asarray(sim_evt, dtype=np.int64))
                kept_tot += len(feat_evt)

        print(f"\nKept {kept_tot} / {init_tot} T5s "
            f"({kept_tot/init_tot*100:.2f} %) that passed the pMatched cut.")
        print(f"Total events with ≥1 kept T5: {len(features_per_event)}")

        # ------------------------------------------------------------------

        print(f"Writing to {FEATURES_T5}")
        with open(FEATURES_T5, "wb") as fi:
            pickle.dump({
                "features_per_event": features_per_event,
                "displaced_per_event": displaced_per_event,
                "sim_indices_per_event": sim_indices_per_event,
            }, fi)

        return features_per_event, displaced_per_event, sim_indices_per_event


    def get_pls_features(self, branches):

        KEEP_FRAC_PLS = 0.40
        print(f"\nBuilding pLS features …")

        n_events = np.shape(branches['pLS_eta'])[0]
        pLS_features_per_event    = []
        pLS_eta_per_event         = []
        pLS_sim_indices_per_event = []

        kept_tot_pls, init_tot_pls = 0, 0
        for ev in tqdm(range(n_events)):
            n_pls = len(branches['pLS_eta'][ev])
            init_tot_pls += n_pls
            if n_pls == 0:
                continue

            feat_evt, eta_evt, sim_evt = [], [], []

            for i in range(n_pls):
                if branches['pLS_isFake'][ev][i]:
                    continue
                if np.random.random() > KEEP_FRAC_PLS:
                    continue

                # ――― hit‑level quantities -------------------------------------------
                eta = branches['pLS_eta'][ev][i]
                etaErr = branches['pLS_etaErr'][ev][i]
                phi = branches['pLS_phi'][ev][i]
                circleCenterX = np.abs(branches['pLS_circleCenterX'][ev][i])
                circleCenterY = np.abs(branches['pLS_circleCenterY'][ev][i])
                circleRadius = branches['pLS_circleRadius'][ev][i]
                ptIn = branches['pLS_ptIn'][ev][i]
                ptErr = branches['pLS_ptErr'][ev][i]
                isQuad = branches['pLS_isQuad'][ev][i]

                # ――― build feature vector -------------------------------------------
                f = [
                    eta/4.0,
                    etaErr/.00139,
                    np.cos(phi),
                    np.sin(phi),
                    1.0 / ptIn,
                    np.log10(ptErr),
                    isQuad,
                    np.log10(circleCenterX),
                    np.log10(circleCenterY),
                    np.log10(circleRadius),
                ]

                feat_evt.append(f)
                eta_evt.append(eta)

                sim_list = branches['pLS_matched_simIdx'][ev][i]
                sim_evt.append(sim_list[0] if len(sim_list) else -1)

            # ――― store per‑event containers -----------------------------------------
            if feat_evt:              # skip events with no survivors
                pLS_features_per_event   .append(np.asarray(feat_evt, dtype=np.float32))
                pLS_eta_per_event        .append(np.asarray(eta_evt,  dtype=np.float32))
                pLS_sim_indices_per_event.append(np.asarray(sim_evt, dtype=np.int64))
                kept_tot_pls += len(feat_evt)

        print(f"\nKept {kept_tot_pls} / {init_tot_pls} pLSs "
            f"({kept_tot_pls/init_tot_pls*100:.2f} %) that passed the selections.")
        print(f"Total events with ≥1 kept pLS: {len(pLS_features_per_event)}")

        # ----------------------------------------------------------------------------

        print(f"Writing to {FEATURES_PLS}")
        with open(FEATURES_PLS, "wb") as fi:
            pickle.dump({
                "pLS_features_per_event": pLS_features_per_event,
                "pLS_sim_indices_per_event": pLS_sim_indices_per_event,
            }, fi)


        return pLS_features_per_event, pLS_sim_indices_per_event


    def get_t5_pairs(self, features_per_event, displaced_per_event, sim_indices_per_event):

        # invoke
        X_left, X_right, y, disp_L, disp_R, true_L, true_R = create_t5_pairs_balanced_parallel(
            features_per_event,
            sim_indices_per_event,
            displaced_per_event,
            max_similar_pairs_per_event    = 1000,
            max_dissimilar_pairs_per_event = 1000,
            invalid_sim_idx                = -1,
            n_workers                      = min(32, os.cpu_count() // 2),
        )

        if len(y) == 0:
            raise ValueError("No pairs generated. Check filters/data.")

        mask = (np.isfinite(X_left).all(axis=1) &
                np.isfinite(X_right).all(axis=1))
        if not mask.all():
            print(f"Filtering {np.sum(~mask)} pairs with NaN/Inf")
            X_left, X_right, y, disp_L, disp_R = X_left[mask], X_right[mask], y[mask], disp_L[mask], disp_R[mask]

        weights_t5 = np.where(disp_L | disp_R, 5.0, 1.0).astype(np.float32)

        X_left_train, X_left_test, \
        X_right_train, X_right_test, \
        y_t5_train, y_t5_test, \
        w_t5_train, w_t5_test, \
        true_L_train, true_L_test, \
        true_R_train, true_R_test = train_test_split(
            X_left, X_right, y, weights_t5, true_L, true_R,
            test_size=0.20, random_state=42,
            stratify=y, shuffle=True
        )

        # compute displaced fraction
        pct_disp = np.mean(disp_L | disp_R) * 100
        print(f"{pct_disp:.2f}% of all pairs involve a displaced T5")

        # write results to file
        print(f"Writing to {PAIRS_T5T5}")
        with open(PAIRS_T5T5, "wb") as fi:
            pickle.dump({
                "X_left_train": X_left_train,
                "X_left_test": X_left_test,
                "X_right_train": X_right_train,
                "X_right_test": X_right_test,
                "y_t5_train": y_t5_train,
                "y_t5_test": y_t5_test,
                "w_t5_train": w_t5_train,
                "w_t5_test": w_t5_test,
                "true_L_train": true_L_train,
                "true_L_test": true_L_test,
                "true_R_train": true_R_train,
                "true_R_test": true_R_test,
            }, fi)

        return [
            X_left_train, X_left_test,
            X_right_train, X_right_test,
            y_t5_train, y_t5_test,
            w_t5_train, w_t5_test,
            true_L_train, true_L_test,
            true_R_train, true_R_test
        ]


    def get_t5_pls_pairs(self, pLS_features_per_event, pLS_sim_indices_per_event, features_per_event, displaced_per_event, sim_indices_per_event):

        # now drive over all events in parallel, with a global timer & totals
        print(f"\n>>> Building pLS-T5 pairs (ΔR² < {DELTA_R2_CUT_PLS_T5}) …")
        t0 = time.time()
        all_packed = []
        sim_total = 0
        dis_total = 0

        n_workers = min(32, os.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    _pairs_pLS_T5_single_vectorized, ev,
                    pLS_features_per_event[ev],
                    pLS_sim_indices_per_event[ev],
                    features_per_event[ev],
                    sim_indices_per_event[ev],
                    displaced_per_event[ev],
                    MAX_SIM, MAX_DIS, INVALID_SIM_IDX
                )
                for ev in range(len(features_per_event))
                # for ev in range(50)
            ]
            for fut in futures:
                _, packed = fut.result()
                # accumulate
                sim_evt = sum(1 for _,_,lbl,_ in packed if lbl == 0)
                dis_evt = sum(1 for _,_,lbl,_ in packed if lbl == 1)
                sim_total += sim_evt
                dis_total += dis_evt
                all_packed.extend(packed)

        print(f"<<< done in {time.time() - t0:.1f}s  "
            f"| sim {sim_total:5d}  dis {dis_total:5d}  total {len(all_packed):,d}")

        # unpack into numpy arrays
        pls_feats = np.array([p[0] for p in all_packed], dtype=np.float32)
        t5_feats  = np.array([p[1] for p in all_packed], dtype=np.float32)
        y_pls     = np.array([p[2] for p in all_packed], dtype=np.int32)
        disp_flag = np.array([p[3] for p in all_packed], dtype=bool)
        w_pls     = np.array([5.0 if p[3] else 1.0 for p in all_packed], dtype=np.float32)

        # train/test split
        X_pls_train, X_pls_test, \
        X_t5raw_train, X_t5raw_test, \
        y_pls_train, y_pls_test, \
        w_pls_train, w_pls_test = train_test_split(
            pls_feats, t5_feats, y_pls, w_pls,
            test_size=0.20, random_state=42,
            stratify=y_pls, shuffle=True
        )

        pct_disp_pls = disp_flag.mean() * 100.0
        print(f"pLS-T5 pairs → train {len(y_pls_train)}  test {len(y_pls_test)}")
        print(f"{pct_disp_pls:.2f}% of pLS-T5 pairs involve a displaced T5")

        print(f"Writing to {PAIRS_T5PLS}")
        with open(PAIRS_T5PLS, "wb") as fi:
            pickle.dump({
                "X_pls_train": X_pls_train,
                "X_pls_test": X_pls_test,
                "X_t5raw_train": X_t5raw_train,
                "X_t5raw_test": X_t5raw_test,
                "y_pls_train": y_pls_train,
                "y_pls_test": y_pls_test,
                "w_pls_train": w_pls_train,
                "w_pls_test": w_pls_test
        }, fi)

        return [
            X_pls_train, X_pls_test,
            X_t5raw_train, X_t5raw_test,
            y_pls_train, y_pls_test,
            w_pls_train, w_pls_test,
        ]


def _pairs_single_event(evt_idx,
                        F, S, D,
                        max_sim, max_dis,
                        invalid_sim):
    """
    Worker run in a separate process.
    Returns two Python lists with the selected (i,j) indices per event.
    """
    t0 = time.time()
    n = F.shape[0]
    if n < 2:
        return evt_idx, [], []

    eta1 = F[:, 0] * ETA_MAX
    phi1 = np.arctan2(F[:, 2], F[:, 1])

    sim_pairs, dis_pairs = [], []

    # similar pairs (same sim-index)
    buckets = {}
    for idx, s in enumerate(S):
        if s != invalid_sim:
            buckets.setdefault(s, []).append(idx)

    for lst in buckets.values():
        if len(lst) < 2:
            continue
        for a in range(len(lst) - 1):
            i = lst[a]
            for b in range(a + 1, len(lst)):
                j = lst[b]
                dphi = delta_phi(phi1[i], phi1[j])
                dr2  = (eta1[i] - eta1[j])**2 + dphi**2
                if dr2 < DELTA_R2_CUT:
                    sim_pairs.append((i, j))

    # dissimilar pairs (different sim)
    for i in range(n - 1):
        si, ei, pi = S[i], eta1[i], phi1[i]
        for j in range(i + 1, n):
            # skip fake-fake pairs
            if si == invalid_sim and S[j] == invalid_sim:
                continue
            if (si == S[j]) and si != invalid_sim:
                continue
            dphi = delta_phi(pi, phi1[j])
            dr2  = (ei - eta1[j])**2 + dphi**2
            if dr2 < DELTA_R2_CUT:
                dis_pairs.append((i, j))

    # down-sample
    if len(sim_pairs) > max_sim:
        sim_pairs = random.sample(sim_pairs, max_sim)
    if len(dis_pairs) > max_dis:
        dis_pairs = random.sample(dis_pairs, max_dis)

    dt = time.time() - t0
    print(f"[evt {evt_idx:4d}]  T5s={n:5d}  sim. pairs={len(sim_pairs):3d}  dis. pairs={len(dis_pairs):3d} | {dt:.1f} seconds")
    return evt_idx, sim_pairs, dis_pairs


def _pairs_single_event_vectorized(evt_idx,
                                   F, S, D,
                                   max_sim, max_dis,
                                   invalid_sim):
    t0 = time.time()
    n = F.shape[0]
    eta1, phi1 = F[:, 0] * ETA_MAX, np.arctan2(F[:, 2], F[:, 1])

    # upper-triangle (non-diagonal) indices
    idx_l, idx_r = np.triu_indices(n, k=1)
    idxs_triu = np.stack((idx_l, idx_r), axis=-1)

    simidx_l = S[idx_l]
    simidx_r = S[idx_r]

    eta_l = eta1[idx_l]
    eta_r = eta1[idx_r]
    phi_l = phi1[idx_l]
    phi_r = phi1[idx_r]
    dphi = np.abs(phi_l - phi_r)
    dphi[dphi > np.pi] -= 2 * np.pi  # adjust to [-pi, pi]
    dr2 = (eta_l - eta_r)**2 + dphi**2

    dr2_valid = (dr2 < DELTA_R2_CUT)
    sim_idx_same = (simidx_l == simidx_r)
    sim_mask = dr2_valid & sim_idx_same & (simidx_l != invalid_sim)
    dis_mask = dr2_valid & ~sim_idx_same

    sim_pairs = idxs_triu[sim_mask]
    dis_pairs = idxs_triu[dis_mask]

    # down-sample
    random.seed(evt_idx)
    if len(sim_pairs) > max_sim:
        sim_pairs = sim_pairs[random.sample(range(len(sim_pairs)), max_sim)]
    if len(dis_pairs) > max_dis:
        dis_pairs = dis_pairs[random.sample(range(len(dis_pairs)), max_dis)]

    dt = time.time() - t0
    print(f"[evt {evt_idx:4d}]  T5s={n:5d}  sim. pairs={len(sim_pairs):3d}  dis. pairs={len(dis_pairs):3d} | {dt:.1f} seconds vectorized")
    return evt_idx, sim_pairs, dis_pairs


def create_t5_pairs_balanced_parallel(features_per_event,
                                      sim_indices_per_event,
                                      displaced_per_event,
                                      *,
                                      max_similar_pairs_per_event=100,
                                      max_dissimilar_pairs_per_event=450,
                                      invalid_sim_idx=-1,
                                      n_workers=None,
                                      vectorize=True):
    t0 = time.time()
    print("\n>>> Pair generation  (ΔR² < 0.02)  –  parallel mode")

    work_args = [
        (evt_idx,
        features_per_event[evt_idx],
        sim_indices_per_event[evt_idx],
        displaced_per_event[evt_idx],
        max_similar_pairs_per_event,
        max_dissimilar_pairs_per_event,
        invalid_sim_idx)
        for evt_idx in range(len(features_per_event))
        # for evt_idx in range(50)
    ]

    func = _pairs_single_event_vectorized if vectorize else _pairs_single_event
    sim_L, sim_R, sim_disp, true_sim_L, true_sim_R = [], [], [], [], []
    dis_L, dis_R, dis_disp, true_dis_L, true_dis_R = [], [], [], [], []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(func, *args) for args in work_args]
        for fut in futures:
            evt_idx, sim_pairs_evt, dis_pairs_evt = fut.result()
            F = features_per_event[evt_idx]
            S = sim_indices_per_event[evt_idx]
            D = displaced_per_event[evt_idx]

            for i, j in sim_pairs_evt:
                sim_L.append(F[i])
                sim_R.append(F[j])
                sim_disp.append(D[i] > 0.1 or D[j] > 0.1)
                true_sim_L.append(S[i])
                true_sim_R.append(S[j])

            for i, j in dis_pairs_evt:
                dis_L.append(F[i])
                dis_R.append(F[j])
                dis_disp.append(D[i] > 0.1 or D[j] > 0.1)
                true_dis_L.append(S[i])
                true_dis_R.append(S[j])

    X_left  = np.concatenate([np.asarray(sim_L, dtype=np.float32),
                              np.asarray(dis_L, dtype=np.float32)], axis=0)
    X_right = np.concatenate([np.asarray(sim_R, dtype=np.float32),
                              np.asarray(dis_R, dtype=np.float32)], axis=0)
    y       = np.concatenate([np.zeros(len(sim_L), dtype=np.int32),
                              np.ones (len(dis_L), dtype=np.int32)])

    disp_L = np.concatenate([np.asarray(sim_disp, dtype=bool),
                             np.asarray(dis_disp, dtype=bool)], axis=0)
    disp_R = disp_L.copy()

    true_L = np.concatenate([np.asarray(true_sim_L, dtype=np.int64),
                             np.asarray(true_dis_L, dtype=np.int64)], axis=0)
    true_R = np.concatenate([np.asarray(true_sim_R, dtype=np.int64),
                             np.asarray(true_dis_R, dtype=np.int64)], axis=0)

    print(f"<<< done in {time.time() - t0:.1f}s  | sim {len(sim_L)}  dis {len(dis_L)}  total {len(y)}")
    return X_left, X_right, y, disp_L, disp_R, true_L, true_R


def _pairs_pLS_T5_single_vectorized(evt_idx,
                                    F_pLS, S_pLS,
                                    F_T5,  S_T5, D_T5,
                                    max_sim, max_dis,
                                    invalid_sim):
    """
    Build similar / dissimilar pLS-T5 pairs for a single event,
    printing per-event summary.
    """
    t0 = time.time()
    n_p, n_t = F_pLS.shape[0], F_T5.shape[0]
    sim_pairs, dis_pairs = [], []

    # if either collection is empty, report zeros and bail
    if n_p == 0 or n_t == 0:
        print(f"[evt {evt_idx:4d}]  pLSs={n_p:5d}  T5s={n_t:5d}  sim={0:4d}  dis={0:4d}")
        return evt_idx, []

    # un-normalize eta and compute phi angles
    eta_p = F_pLS[:,0] * 4.0
    phi_p = np.arctan2(F_pLS[:,3], F_pLS[:,2])
    eta_t = F_T5[:,0] * ETA_MAX
    phi_t = np.arctan2(F_T5[:,2], F_T5[:,1])

    # make all possible pairs (i, j)
    idx_p, idx_t = np.indices( (n_p, n_t) )
    idx_p, idx_t = idx_p.flatten(), idx_t.flatten()

    # calculate angles
    dphi = (phi_p[idx_p] - phi_t[idx_t] + np.pi) % (2 * np.pi) - np.pi
    dr2 = (eta_p[idx_p] - eta_t[idx_t])**2 + dphi**2
    dr2_valid = (dr2 < DELTA_R2_CUT_PLS_T5)

    # compare sim indices
    simidx_p = S_pLS[idx_p]
    simidx_t = S_T5[idx_t]
    sim_idx_same = (simidx_p == simidx_t)

    # create masks for similar and dissimilar pairs
    sim_mask = dr2_valid & sim_idx_same & (simidx_p != invalid_sim)
    dis_mask = dr2_valid & ~sim_idx_same

    # get the pairs
    sim_pairs = np.column_stack((idx_p[sim_mask], idx_t[sim_mask]))
    dis_pairs = np.column_stack((idx_p[dis_mask], idx_t[dis_mask]))

    # down-sample
    random.seed(evt_idx)
    if len(sim_pairs) > max_sim:
        sim_pairs = sim_pairs[random.sample(range(len(sim_pairs)), max_sim)]
    if len(dis_pairs) > max_dis:
        dis_pairs = dis_pairs[random.sample(range(len(dis_pairs)), max_dis)]

    # print per-event summary
    dt = time.time() - t0
    print(f"[evt {evt_idx:4d}]  pLSs={n_p:5d}  T5s={n_t:5d}  "
          f"sim. pairs={len(sim_pairs):4d}  dis. pairs={len(dis_pairs):4d} | {dt:.1f} seconds vectorized")


    # pack into (feature, feature, label, displaced_flag)
    packed = []
    for i,j in sim_pairs:
        packed.append((F_pLS[i], F_T5[j], 0, D_T5[j] > DISP_VXY_CUT))
    for i,j in dis_pairs:
        packed.append((F_pLS[i], F_T5[j], 1, D_T5[j] > DISP_VXY_CUT))

    return evt_idx, packed


def _pairs_pLS_T5_single(evt_idx,
                         F_pLS, S_pLS,
                         F_T5,  S_T5, D_T5,
                         max_sim, max_dis,
                         invalid_sim):
    """
    Build similar / dissimilar pLS-T5 pairs for a single event,
    printing per-event summary.
    """
    t0 = time.time()
    n_p, n_t = F_pLS.shape[0], F_T5.shape[0]
    sim_pairs, dis_pairs = [], []

    # if either collection is empty, report zeros and bail
    if n_p == 0 or n_t == 0:
        print(f"[evt {evt_idx:4d}]  pLSs={n_p:5d}  T5s={n_t:5d}  sim={0:4d}  dis={0:4d}")
        return evt_idx, []

    # un-normalize eta and compute phi angles
    eta_p = F_pLS[:,0] * 4.0
    phi_p = np.arctan2(F_pLS[:,3], F_pLS[:,2])
    eta_t = F_T5[:,0] * ETA_MAX
    phi_t = np.arctan2(F_T5[:,2], F_T5[:,1])

    # bucket T5 by sim-idx for similar
    buckets = {}
    for j,s in enumerate(S_T5):
        if s != invalid_sim:
            buckets.setdefault(s, []).append(j)
    for i,s in enumerate(S_pLS):
        if s == invalid_sim:
            continue
        for j in buckets.get(s, []):
            dphi = (phi_p[i] - phi_t[j] + np.pi) % (2*np.pi) - np.pi
            dr2  = (eta_p[i] - eta_t[j])**2 + dphi**2
            if dr2 < DELTA_R2_CUT_PLS_T5:
                sim_pairs.append((i,j))

    # find dissimilar (different sim-idx) pairs
    for i in range(n_p):
        for j in range(n_t):
            if S_pLS[i] == S_T5[j] and S_pLS[i] != invalid_sim:
                continue
            dphi = (phi_p[i] - phi_t[j] + np.pi) % (2*np.pi) - np.pi
            dr2  = (eta_p[i] - eta_t[j])**2 + dphi**2
            if dr2 < DELTA_R2_CUT_PLS_T5:
                dis_pairs.append((i,j))

    # down-sample to limits
    if len(sim_pairs) > max_sim:
        sim_pairs = random.sample(sim_pairs, max_sim)
    if len(dis_pairs) > max_dis:
        dis_pairs = random.sample(dis_pairs, max_dis)

    # print per-event summary
    dt = time.time() - t0
    print(f"[evt {evt_idx:4d}]  pLSs={n_p:5d}  T5s={n_t:5d}  "
          f"sim. pairs={len(sim_pairs):4d}  dis. pairs={len(dis_pairs):4d} | {dt:.1f} seconds")

    # pack into (feature, feature, label, displaced_flag)
    packed = []
    for i,j in sim_pairs:
        packed.append((F_pLS[i], F_T5[j], 0, D_T5[j] > DISP_VXY_CUT))
    for i,j in dis_pairs:
        packed.append((F_pLS[i], F_T5[j], 1, D_T5[j] > DISP_VXY_CUT))

    return evt_idx, packed
