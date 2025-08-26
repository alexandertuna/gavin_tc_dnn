import argparse
import numpy as np
import pandas as pd
from io import StringIO
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams.update({'font.size': 16})

FNAME = "/ceph/users/atuna/work/gavin_tc_dnn/experiments/phi_dependence_on_dr2/phi_correlation_2.txt"
LINE_TAG = "phi_correlation"

def main():

    df = get_dataframe(FNAME)
    print(df.head())
    print(df["comparison"] == 't5t5')

    with PdfPages("phi_dependence_on_dr2.pdf") as pdf:
        plot(df, pdf)


def plot(df: pd.DataFrame, pdf: PdfPages) -> None:
    for status in [0, 1]:
        dup = "duplicates" if status == 0 else "non-duplicates"
        for phi_used in ["norm"]:
            for comparison in ["t5t5", "plspls"]:
                for corr in ["pearson", "spearman"]:
                    mask = (df["status"] == status) & (df["comparison"] == comparison) & (df["phi_used"] == phi_used)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(df["dr2"][mask],
                               df[corr][mask],
                               label=corr.capitalize(),
                               s=100,
                               )
                    ax.set_xlabel(r"Max $\Delta R^{2}$ for track pairs")
                    ax.set_ylabel(f"{corr.capitalize()} correlation")
                    ax.set_ylim([-1, 1])
                    ax.set_title(f"{comparison.upper()} {dup}")
                    ax.grid(True)
                    ax.tick_params(right=True, top=True)
                    ax.set_axisbelow(True)
                    fig.subplots_adjust(right=0.98, left=0.16, bottom=0.09, top=0.95)
                    pdf.savefig()
                    plt.close()


def get_dataframe(fname: str) -> pd.DataFrame:
    """
    phi_correlation='./dr2_proj_0.6'
    phi_correlation=0.437230; spearman=0.26347658; excluded='19%'; comparison='t5t5'; status=0; dim=-1; feature=31;
    phi_correlation=0.913274; spearman=0.90507692; excluded='0%'; comparison='t5t5'; status=1; dim=-1; feature=31;
    phi_correlation=0.910048; spearman=0.88046681; excluded='0%'; comparison='plspls'; status=0; dim=-1; feature=11;
    phi_correlation=0.957695; spearman=0.95045852; excluded='0%'; comparison='plspls'; status=1; dim=-1; feature=11;
    """
    rows = []
    with open(fname, 'r') as fi:
        phi_used, dr2 = None, None
        for line in fi:
            if not line.startswith(LINE_TAG):
                continue
            line = line.strip()
            if "dr2_" in line:
                experiment = line.split("=")[1].replace("'", "")
                experiment = experiment.split("_")
                phi_used = experiment[-2]
                dr2 = float(experiment[-1])
            else:
                line = line.replace(" ", "")
                line = line.rstrip(";")
                kvs = line.split(";")
                row = {k: v for (k, v) in (x.split("=") for x in kvs)}
                row["phi_used"] = phi_used
                row["dr2"] = dr2
                row["spearman"] = float(row["spearman"])
                row["pearson"] = float(row["phi_correlation"])
                row["status"] = int(row["status"])
                row["comparison"] = row["comparison"].strip("'")
                rows.append(row)

    print(len(rows))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

