import awkward as ak
import numpy as np
import uproot
from pathlib import Path
from collections import Counter

LSTNTUPLE = Path("/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75.root")
BRANCH = "pLS_matched_simIdx"

def main():
    data = get_data()
    simidxs_all = data[BRANCH]
    print(f"Inspecting {BRANCH} ...")

    nev = len(simidxs_all)

    for ev in range(nev):
        simidxs = ak.flatten(simidxs_all[ev]).to_numpy()
        count = Counter(simidxs)
        most_common = count.most_common(3)
        unique_simidxs = np.unique(simidxs)
        print(f"Event {ev}: {len(simidxs)} total simIdxs, {len(unique_simidxs)} unique simIdxs")
        for simidx, freq in most_common:
            print(f"  simIdx {simidx}: {freq} occurrences")

        if ev > 10:
            break

def get_data():
    with uproot.open(f"{LSTNTUPLE}:tree") as tr:
        return tr.arrays([BRANCH])

if __name__ == "__main__":
    main()
