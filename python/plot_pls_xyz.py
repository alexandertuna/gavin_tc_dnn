import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams["font.size"] = 16


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--input", type=str, default="/Users/alexandertuna/Downloads/cms/gavin_tc_dnn/data/log_1_event_pls_mod.txt",
    #                     help="Input LSTNtuple ROOT file")
    parser.add_argument("--input", type=str, default="/Users/alexandertuna/Downloads/cms/gavin_tc_dnn/data/log_1_event_LSTPrepareInput.txt",
                        help="Input LSTNtuple ROOT file")
    parser.add_argument("--pdf", type=str, default="pls_xyz.pdf",
                        help="Path to save the output plots in PDF format")
    return parser.parse_args()


def main():
    args = options()

    if "pls_mod.txt" in args.input:

        cols = ["1x", "1y", "1z",
                "2x", "2y", "2z",
                "3x", "3y", "3z",
                "4x", "4y", "4z"]
        df = pd.read_csv(args.input, sep=",", names=cols)
        df["1r"] = np.sqrt(df["1x"]**2 + df["1y"]**2)
        df["2r"] = np.sqrt(df["2x"]**2 + df["2y"]**2)
        df["3r"] = np.sqrt(df["3x"]**2 + df["3y"]**2)
        df["4r"] = np.sqrt(df["4x"]**2 + df["4y"]**2)
        print(df)

        with PdfPages(args.pdf) as pdf:
            plot_xy(args.input, df, pdf)
            plot_rz(args.input, df, pdf)

    elif "all.txt" in args.input or "LSTPrepareInput.txt" in args.input:

        cols = ["x", "y", "z", "detid"]
        df = pd.read_csv(args.input, sep=",", names=cols)
        df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)
        print(df)

        with PdfPages(args.pdf) as pdf:
            plot_xy_all(args.input, df, pdf)
            plot_rz_all(args.input, df, pdf)


def plot_xy(fname: str, df: pd.DataFrame, pdf: PdfPages) -> None:
    zmax, rmax = 20, 20
    colors = {"1": "r", "2": "g", "3": "b", "4": "m"}
    mask = (np.abs(df["1z"]) < zmax) & (np.abs(df["2z"]) < zmax) & (np.abs(df["3z"]) < zmax) & (np.abs(df["4z"]) < zmax)
    subset = df[mask]
    fig, ax = plt.subplots(figsize=(8, 8))
    # ax.scatter(subset["1x"], subset["1y"], s=1, c=colors["1"], label="Hit 1")
    # ax.scatter(subset["2x"], subset["2y"], s=1, c=colors["2"], label="Hit 2")
    ax.scatter(subset["3x"], subset["3y"], s=1, c=colors["3"], label="Hit 3")
    # ax.scatter(subset["4x"], subset["4y"], s=1, c=colors["4"], label="Hit 4")
    ax.set_title(os.path.basename(fname))
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.text(0.1, 0.9, f"|z| < {zmax} cm", transform=ax.transAxes)
    ax.legend()
    fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.1)
    pdf.savefig()
    plt.close()


def plot_rz(fname: str, df: pd.DataFrame, pdf: PdfPages) -> None:
    zmax, rmax = 20, 20
    colors = {"1": "r", "2": "g", "3": "b", "4": "m"}
    mask = (np.abs(df["1z"]) < zmax) & (np.abs(df["2z"]) < zmax) & (np.abs(df["3z"]) < zmax) & (np.abs(df["4z"]) < zmax)
    subset = df[mask]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(subset["4z"], subset["4r"], s=1, c=colors["4"], label="Hit 4")
    ax.scatter(subset["3z"], subset["3r"], s=1, c=colors["3"], label="Hit 3")
    ax.scatter(subset["2z"], subset["2r"], s=1, c=colors["2"], label="Hit 2")
    ax.scatter(subset["1z"], subset["1r"], s=1, c=colors["1"], label="Hit 1")
    ax.set_title(os.path.basename(fname))
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("r [cm]")
    ax.set_xlim(-zmax, zmax)
    ax.set_ylim(0, rmax)
    ax.text(0.1, 0.9, f"|z| < {zmax} cm", transform=ax.transAxes)
    ax.legend()
    fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.1)
    pdf.savefig()
    plt.close()


def plot_xy_all(fname: str, df: pd.DataFrame, pdf: PdfPages) -> None:
    zmax, rmax = 20, 20
    mask = (np.abs(df["z"]) < zmax) & (df["detid"] == 1)
    subset = df[mask]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(subset["x"], subset["y"], s=1, c="r")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.text(0.1, 0.9, f"|z| < {zmax} cm", transform=ax.transAxes)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.1)
    pdf.savefig()
    plt.close()


def plot_rz_all(fname: str, df: pd.DataFrame, pdf: PdfPages) -> None:
    zmax, rmax = 20, 20
    mask = (np.abs(df["z"]) < zmax) & (df["detid"] == 1)
    subset = df[mask]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(subset["z"], subset["r"], s=1, c="r")
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("r [cm]")
    ax.set_xlim(-zmax, zmax)
    ax.set_ylim(0, rmax)
    ax.text(0.1, 0.9, f"|z| < {zmax} cm", transform=ax.transAxes)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.1)
    pdf.savefig()
    plt.close()


if __name__ == "__main__":
    main()
