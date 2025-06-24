"""
Data looks like:
Result vectorize=True n_workers=256 it=0 174.1208 seconds
Result vectorize=True n_workers=256 it=1 64.9057 seconds
Result vectorize=True n_workers=256 it=2 50.7655 seconds
Result vectorize=True n_workers=192 it=0 40.6873 seconds
Result vectorize=True n_workers=192 it=1 40.9738 seconds
Result vectorize=True n_workers=192 it=2 40.9859 seconds
Result vectorize=True n_workers=128 it=0 31.1482 seconds
Result vectorize=True n_workers=128 it=1 32.7579 seconds
Result vectorize=True n_workers=128 it=2 35.3546 seconds
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import socket

rcParams["font.size"] = 16
rcParams["axes.labelpad"] = 10

FPATH = Path("workers3.txt")
PDFPATH = Path("parallel_test.pdf")
COLORS = ["green", "blue"]
LABELS = ["For-loop", "Vectorized"]

def main():
    data = get_data()
    median_data = get_median_data(data)
    with PdfPages(PDFPATH) as pdf:
        plot_data(median_data, pdf)


def get_data() -> np.array:
    raw_data = []
    with open(FPATH, "r") as fi:
        for line in fi:
            if "Result" in line:
                result, vectorize, n_workers, it, elapsed, seconds = line.split()
                vectorize = eval(vectorize.split("=")[1])
                n_workers = int(n_workers.split("=")[1])
                it = int(it.split("=")[1])
                elapsed = float(elapsed)
                raw_data.append((vectorize, n_workers, it, elapsed))
    return np.array(raw_data)


def get_median_data(data: np.array) -> np.array:
    # take median elapsed time for each vectorize and n_workers combination
    median_data = []
    for vectorize in np.unique(data[:, 0]):
        for n_workers in np.unique(data[:, 1]):
            subset = data[(data[:, 0] == vectorize) & (data[:, 1] == n_workers)]
            if len(subset) == 0:
                continue
            median_elapsed = np.median(subset[:, 3].astype(float))
            median_data.append((vectorize, n_workers, 0, median_elapsed))
    return np.array(median_data)    


def plot_data(data: np.array, pdf: PdfPages):
    fig, ax = plt.subplots(figsize=(8, 8))
    for vectorize in np.unique(data[:, 0]):
        subset = data[data[:, 0] == vectorize]
        n_workers = subset[:, 1].astype(int)
        elapsed = subset[:, 3].astype(float)
        ax.plot(n_workers, elapsed, marker='o', color=COLORS[int(vectorize)])

    # put ratio of elapsed time for vectorized vs for-loop on second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel("Ratio of For-loop / Vectorized", color="purple")
    n_workers = np.unique(data[:, 1])
    elapsed_for_loop = data[data[:, 0] == False][:, 3].astype(float)
    elapsed_vectorized = data[data[:, 0] == True][:, 3].astype(float)
    if len(elapsed_for_loop) > 0 and len(elapsed_vectorized) > 0:
        n_missing = len(n_workers) - len(elapsed_for_loop)
        elapsed_for_loop = np.pad(elapsed_for_loop, (n_missing, 0), constant_values=0)
        ratio = elapsed_for_loop / elapsed_vectorized
        ax2.plot(n_workers, ratio, marker='o', color="purple")

    # legend
    xtext, ytext, dy = 0.62, 0.75, 0.05
    for color, label in zip(COLORS, LABELS):
        ax.text(xtext, ytext - dy * COLORS.index(color), label, color=color, transform=ax.transAxes, fontsize=20)
    ax.text(xtext, ytext - dy * len(COLORS), "Ratio", color="purple", transform=ax.transAxes, fontsize=20)

    # make it pretty
    ax.set_xlabel("Number of threads (ThreadPoolExecutor)")
    ax.set_ylabel("Elapsed Time (median seconds, 3 trials)")
    ax.set_title(f"Processing time vs threads on {socket.gethostname()}")
    ax.set_ylim(0, None)
    ax.grid(True)
    ax.tick_params(right=True, top=True, direction="in")
    fig.subplots_adjust(bottom=0.10, left=0.15, right=0.90, top=0.94)
    pdf.savefig(fig)
    plt.close(fig)

    # zoom and save
    ax.set_ylim(0, 250)
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
