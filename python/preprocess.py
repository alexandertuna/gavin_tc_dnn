import os
import pickle
import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import awkward as ak # Using awkward array for easier handling of jagged data
import time # For timing steps

import time, random, math, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

DELTA_R2_CUT = 0.02
ETA_MAX = 2.5

# pairing hyper-parameters
DELTA_R2_CUT_PLS_T5 = 0.02
DISP_VXY_CUT       = 0.1
INVALID_SIM_IDX    = -1
MAX_SIM            = 1000
MAX_DIS            = 1000


def load_root_file(file_path, branches=None, print_branches=False):
    all_branches = {}
    with uproot.open(file_path) as file:
        tree = file["tree"]
        # Load all ROOT branches into array if not specified
        if branches is None:
            branches = tree.keys()
        # Option to print the branch names
        if print_branches:
            print("Branches:", tree.keys())
        # Each branch is added to the dictionary
        for branch in branches:
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


class Preprocessor:
    def __init__(self, root_path):
        self.load_root_file(root_path)

    def load_root_file(self, root_path):

        print("Loading ROOT file:", root_path)
        branches = load_root_file(root_path, branches_list, print_branches=True)

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
        for ev in range(n_events):

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

        print("Writing to 0.pkl!")
        with open("0.pkl", "wb") as fi:
            pickle.dump({
                "features_per_event": features_per_event,
                "displaced_per_event": displaced_per_event,
                "sim_indices_per_event": sim_indices_per_event,
            }, fi)

        # ------------------------------------------------------------------

        KEEP_FRAC_PLS = 0.40
        print(f"\nBuilding pLS features …")

        pLS_features_per_event    = []
        pLS_eta_per_event         = []
        pLS_sim_indices_per_event = []

        kept_tot_pls, init_tot_pls = 0, 0
        for ev in range(n_events):
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

        print("Writing to 1.pkl!")
        with open("1.pkl", "wb") as fi:
            pickle.dump({
                "pLS_features_per_event": pLS_features_per_event,
                "pLS_sim_indices_per_event": pLS_sim_indices_per_event,
            }, fi)


        # ----------------------------------------------------------------------------

        # invoke
        X_left, X_right, y, disp_L, disp_R = create_t5_pairs_balanced_parallel(
            features_per_event,
            sim_indices_per_event,
            displaced_per_event,
            max_similar_pairs_per_event    = 1000,
            max_dissimilar_pairs_per_event = 1000,
            invalid_sim_idx                = -1,
            n_workers                      = None
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
        w_t5_train, w_t5_test = train_test_split(
            X_left, X_right, y, weights_t5,
            test_size=0.20, random_state=42,
            stratify=y, shuffle=True
        )

        # compute displaced fraction
        pct_disp = np.mean(disp_L | disp_R) * 100
        print(f"{pct_disp:.2f}% of all pairs involve a displaced T5")

        # ----------------------------------------------------------------------------

        # now drive over all events in parallel, with a global timer & totals
        print(f"\n>>> Building pLS-T5 pairs (ΔR² < {DELTA_R2_CUT_PLS_T5}) …")
        t0 = time.time()
        all_packed = []
        sim_total = 0
        dis_total = 0

        with ProcessPoolExecutor() as pool:
            futures = [
                pool.submit(
                    _pairs_pLS_T5_single, ev,
                    pLS_features_per_event[ev],
                    pLS_sim_indices_per_event[ev],
                    features_per_event[ev],
                    sim_indices_per_event[ev],
                    displaced_per_event[ev],
                    MAX_SIM, MAX_DIS, INVALID_SIM_IDX
                )
                for ev in range(len(features_per_event))
            ]
            for fut in as_completed(futures):
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



def _pairs_single_event(evt_idx,
                        F, S, D,
                        max_sim, max_dis,
                        invalid_sim):
    n = F.shape[0]
    eta1, phi1 = F[:, 0] * ETA_MAX, np.arctan2(F[:, 2], F[:, 1])

    idx_l, idx_r = np.triu_indices(n, k=1) # k=1 ensures idx_l < idx_r
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
    if len(sim_pairs) > max_sim:
        sim_pairs = sim_pairs[np.random.choice(len(sim_pairs), max_sim, replace=False)]
    if len(dis_pairs) > max_dis:
        dis_pairs = dis_pairs[np.random.choice(len(dis_pairs), max_dis, replace=False)]

    print(f"[evt {evt_idx:4d}]  T5s={n:5d}  sim={len(sim_pairs)}  dis={len(dis_pairs)}")
    return evt_idx, sim_pairs, dis_pairs


def create_t5_pairs_balanced_parallel(features_per_event,
                                    sim_indices_per_event,
                                    displaced_per_event,
                                    *,
                                    max_similar_pairs_per_event=100,
                                    max_dissimilar_pairs_per_event=450,
                                    invalid_sim_idx=-1,
                                    n_workers=None):
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
    ]

    sim_L, sim_R, sim_disp = [], [], []
    dis_L, dis_R, dis_disp = [], [], []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_pairs_single_event, *args) for args in work_args]
        for fut in as_completed(futures):
            evt_idx, sim_pairs_evt, dis_pairs_evt = fut.result()
            F = features_per_event[evt_idx]
            D = displaced_per_event[evt_idx]

            for i, j in sim_pairs_evt:
                sim_L.append(F[i])
                sim_R.append(F[j])
                sim_disp.append(D[i] > 0.1 or D[j] > 0.1)

            for i, j in dis_pairs_evt:
                dis_L.append(F[i])
                dis_R.append(F[j])
                dis_disp.append(D[i] > 0.1 or D[j] > 0.1)

    X_left  = np.concatenate([np.asarray(sim_L, dtype=np.float32),
                            np.asarray(dis_L, dtype=np.float32)], axis=0)
    X_right = np.concatenate([np.asarray(sim_R, dtype=np.float32),
                            np.asarray(dis_R, dtype=np.float32)], axis=0)
    y       = np.concatenate([np.zeros(len(sim_L), dtype=np.int32),
                            np.ones (len(dis_L), dtype=np.int32)])

    disp_L = np.concatenate([np.asarray(sim_disp, dtype=bool),
                            np.asarray(dis_disp, dtype=bool)], axis=0)
    disp_R = disp_L.copy()

    print(f"<<< done in {time.time() - t0:.1f}s  | sim {len(sim_L)}  dis {len(dis_L)}  total {len(y)}")
    return X_left, X_right, y, disp_L, disp_R


def _pairs_pLS_T5_single(evt_idx,
                         F_pLS, S_pLS,
                         F_T5,  S_T5, D_T5,
                         max_sim, max_dis,
                         invalid_sim):
    """
    Build similar / dissimilar pLS-T5 pairs for a single event,
    printing per-event summary.
    """
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
    print(f"[evt {evt_idx:4d}]  pLSs={n_p:5d}  T5s={n_t:5d}  "
          f"sim={len(sim_pairs):4d}  dis={len(dis_pairs):4d}")

    # pack into (feature, feature, label, displaced_flag)
    packed = []
    for i,j in sim_pairs:
        packed.append((F_pLS[i], F_T5[j], 0, D_T5[j] > DISP_VXY_CUT))
    for i,j in dis_pairs:
        packed.append((F_pLS[i], F_T5[j], 1, D_T5[j] > DISP_VXY_CUT))

    return evt_idx, packed
