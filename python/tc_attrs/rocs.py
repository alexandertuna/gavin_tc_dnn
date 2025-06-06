import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams.update({'font.size': 20})
from pathlib import Path
import pickle

DIRNAME = "rocs"

def main():
    data = AllTheRocData(DIRNAME)
    plot = PlotRocData(data)


class PlotRocData:
    def __init__(self, data):
        self.data = data
        self.plot()

    def plot(self) -> None:

        fig, ax = plt.subplots(figsize=(8, 8))

        def one_plot(idx: int):
            epoch = self.data.keys[idx]
            train = self.data.data["train"][epoch]
            test = self.data.data["test"][epoch]
            deltaR = self.data.data["deltaR"]

            fpr_train, tpr_train = train["fpr"], train["tpr"]
            fpr_test, tpr_test = test["fpr"], test["tpr"]
            fpr_dr, tpr_dr = deltaR["fpr"], deltaR["tpr"]

            # epoch_data = self.data.data[epoch]
            # fpr_nn, tpr_nn  = epoch_data["fpr_nn"], epoch_data["tpr_nn"]
            # fpr_dr, tpr_dr  = epoch_data["fpr_dr"], epoch_data["tpr_dr"]

            # fig, ax = plt.subplots(figsize=(8, 8))
            ax.clear()
            ptrain = ax.plot(fpr_train, tpr_train, label='Embedding (train)')
            ptest = ax.plot(fpr_test, tpr_test, label='Embedding (test)')
            pdr = ax.plot(fpr_dr, tpr_dr, label=r'$\Delta$R')
            ax.set_xlabel('Duplicate efficiency (False Positive Rate)')
            ax.set_ylabel('Non-duplicate efficiency (True Positive Rate)')
            ax.set_title(f'ROC Curves, epoch {int(epoch)} / {len(self.data.keys)}')
            ax.semilogx()
            ax.set_ylim([0.96, 1.0])
            ax.set_xlim([1e-5, 1])
            ax.legend(loc=(0.50, 0.05))
            ax.grid(True)
            ax.tick_params(right=True)
            ax.tick_params(top=True, which="minor")
            fig.subplots_adjust(left=0.16, right=0.95, top=0.95, bottom=0.13)
            # fig.savefig(f"roc_epoch_{epoch}.pdf")
            return pdr + ptrain + ptest
        
        ani = animation.FuncAnimation(fig, one_plot, frames=len(self.data.keys), interval=1000, blit=True)
        ani.save("rocs.gif", fps=30)


class AllTheRocData:
    def __init__(self, dirname):
        self.dirname = dirname
        self.data = self.load_data()
        self.keys = list(self.data["test"].keys())

    def load_data(self) -> dict:
        data = {}
        data["train"] = {}
        data["test"] = {}
        for path in sorted(Path(self.dirname).rglob("*.npz")):
            if "deltaR" in path.stem:
                data["deltaR"] = np.load(path)
            else:
                epoch = self.path_to_epoch(path)
                sample = "train" if "train" in path.stem else "test"
                data[sample][epoch] = np.load(path)
                # data[epoch] = np.load(fi)
        # for path in sorted(Path(self.dirname).rglob("*.pkl")):
        #     with open(path, "rb") as fi:
        #         epoch = self.path_to_epoch(path)
        #         data[epoch] = pickle.load(fi)
        self.check(data)
        self.announce(data)
        return data

    def path_to_epoch(self, path: Path) -> str:
        stem = path.stem
        stem = stem.replace("roc_", "")
        stem = stem.replace("train_", "")
        stem = stem.replace("test_", "")
        stem = stem.replace("epoch", "")
        return stem

    def check(self, data: dict) -> None:
        keys = list(data.keys())
        if not keys == ["train", "test", "deltaR"]:
            print("Warning: expected keys are ['train', 'test', 'deltaR']")
            print(f"Found keys: {keys}")
            raise ValueError("Unexpected keys in data dictionary")
        train_keys = list(data["train"].keys())
        test_keys = list(data["test"].keys())
        if train_keys != test_keys:
            print(f"Train keys: {train_keys}")
            print(f"Test keys: {test_keys}")
            raise ValueError("Train and test keys do not match")

    def announce(self, data: dict) -> None:
        keys = list(data.keys())
        test_keys = list(data["train"].keys())
        print("Gathered ROC data")
        print(f"Data keys: {", ".join(keys)}")
        print(f"Epochs: {", ".join(test_keys)}")
        print(f"Epoch keys: {list(data["train"][test_keys[0]].keys())}")

if __name__ == "__main__":
    main()
