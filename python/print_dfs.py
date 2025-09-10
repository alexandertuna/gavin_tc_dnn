"""
"""

import argparse
import pickle
import awkward as ak 
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 6)

BONUS_FEATURES = 3

def main():
    args = options()
    (features_per_event,
     displaced_per_event,
     sim_indices_per_event) = load_t5_features(args.features_t5)
    features_per_event = np.concatenate(features_per_event)
    features_per_event = features_per_event[:, :-BONUS_FEATURES]
    
    sim_features_per_event = load_sim_features_unnormalized(args.output_t5)
    print(sim_features_per_event.shape)

    df_feat = pd.DataFrame(features_per_event, columns=feature_names())
    df_sim_feat = pd.DataFrame(sim_features_per_event, columns=["q/pt", "eta", "cos(phi)", "sin(phi)", "pca_dxy", "pca_dz"])

    print("T5 Features:")
    print(df_feat.head())
    print("")
    print("T5 sim Features:")
    print(df_sim_feat.head())
    print("")


def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--features_t5", type=str, default="features_t5.pkl",
                        help="Path to the precomputed T5 features file")
    parser.add_argument("--output_t5", type=str, default="sim_features_t5.pkl",
                        help="Path to save the output T5 features")
    return parser.parse_args()


def load_t5_features(features_t5):
    print(f"Loading T5 features from {features_t5} ...")
    with open(features_t5, "rb") as fi:
        data = pickle.load(fi)
    return [
        data["features_per_event"],
        data["displaced_per_event"],
        data["sim_indices_per_event"],
    ]

def load_sim_features_unnormalized(fname: str):
    feats = load_sim_features(fname)
    sim_pt = np.concatenate(feats["sim_pt"])
    sim_eta = np.concatenate(feats["sim_eta"])
    sim_phi = np.concatenate(feats["sim_phi"])
    sim_pca_dxy = np.concatenate(feats["sim_pca_dxy"])
    sim_pca_dz = np.concatenate(feats["sim_pca_dz"])
    sim_q = np.concatenate(feats["sim_q"])
    return np.column_stack((
        sim_q / sim_pt,
        sim_eta,
        np.cos(sim_phi),
        np.sin(sim_phi),
        sim_pca_dxy,
        sim_pca_dz,
    ))


def load_sim_features(fname: str):
    with open(fname, "rb") as fi:
        data = pickle.load(fi)
        return data["sim_features"]


def feature_names():
    return [
            "eta1 / 2.5",
            "cos(phi1)",
            "sin(phi1)",
            "z1 / z_max",
            "r1 / r_max",

            "eta2 - abs(eta1)",
            "delta_phi(phi2, phi1)",
            "(z2 - z1) / z_max",
            "(r2 - r1) / r_max",

            "eta3 - eta2",
            "delta_phi(phi3, phi2)",
            "(z3 - z2) / z_max",
            "(r3 - r2) / r_max",

            "eta4 - eta3",
            "delta_phi(phi4, phi3)",
            "(z4 - z3) / z_max",
            "(r4 - r3) / r_max",

            "eta5 - eta4",
            "delta_phi(phi5, phi4)",
            "(z5 - z4) / z_max",
            "(r5 - r4) / r_max",

            "1.0 / inR",
            "1.0 / brR",
            "1.0 / outR",

            "s1_fake", "s1_prompt", "s1_disp",
            "d_fake",  "d_prompt",  "d_disp"
        ]

if __name__ == "__main__":
    main()
