from pathlib import Path
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams["font.size"] = 16

from preprocess import load_t5_t5_pairs
from ml import EmbeddingNetT5, EmbeddingNetpLS

def main():
    network = Path("model_weights.pth")
    # network = Path("model_weights_ge0p50.pth")
    plotter = PlotRealAndFakeScores()
    plotter.get_data()
    plotter.get_network(network)
    plotter.get_scores()
    pdf_path = Path("real_fake_scores.pdf")
    with PdfPages(pdf_path) as pdf:
        plotter.plot_scores(pdf)

class PlotRealAndFakeScores:

    def __init__(self):
        self.embed_t5 = EmbeddingNetT5()
        self.embed_pls = EmbeddingNetpLS()


    def get_data(self):
        print("Getting T5-T5 pairs ...")
        (self.X_left_train,
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
         ) = load_t5_t5_pairs()
        print(f"Found {len(self.X_left_train)} training pairs")
        print(f"Found {len(self.X_left_test)} test pairs")
        print(f"Found {len(self.true_L_test)} test pairs")


    def get_network(self, path: Path):
        print(f"Loading model from {path} ...")
        checkpoint = torch.load(path)
        self.embed_t5.load_state_dict(checkpoint["embed_t5"])
        self.embed_pls.load_state_dict(checkpoint["embed_pls"])
        self.embed_t5.eval()
        self.embed_pls.eval()

    
    def get_scores(self):
        print("Calculating T5-T5 scores ...")
        with torch.no_grad():
            x_left = torch.tensor(self.X_left_test)
            x_right = torch.tensor(self.X_right_test)
            e_l = self.embed_t5(x_left)
            e_r = self.embed_t5(x_right)
            self.d = torch.sqrt(((e_l - e_r) ** 2).sum(dim=1, keepdim=True) + 1e-6)
        self.d = self.d.numpy().flatten()


    def plot_scores(self, pdf: PdfPages):
        print(len(self.y_t5_test))
        print(self.true_L_test.shape)
        print(self.true_R_test.shape)
        print(self.true_L_test[self.true_L_test < 0].sum() / len(self.true_L_test))
        print(self.true_R_test[self.true_R_test < 0].sum() / len(self.true_R_test))
        print(self.d.shape)

        bins = np.arange(-0.2, 6.5, 0.1)
        labels = {
            0: "duplicates",
            1: "non-duplicates",
        }
        for y_value in labels:
            fig, ax = plt.subplots(figsize=(8, 8))
            status = self.y_t5_test == y_value
            both_real = (self.true_L_test >= 0) & (self.true_R_test >= 0)
            either_fake = (self.true_L_test < 0) | (self.true_R_test < 0)
            ax.hist(self.d[status & both_real], bins=bins, alpha=0.5, label="Both T5s truth-matched", color="blue")
            ax.hist(self.d[status & either_fake], bins=bins, alpha=0.5, label="One unmatched T5", color="red")
            ax.set_title(f"T5-T5 {labels[y_value]}")
            ax.set_xlabel("Embedding distance")
            ax.set_ylabel("Pairs of tracks")
            ax.tick_params(right=True, top=True, which="both", direction="in")
            fig.subplots_adjust(bottom=0.12, left=0.16, right=0.96, top=0.94)
            ax.legend()

            pdf.savefig()
            ax.semilogy()
            pdf.savefig()
            plt.close()


if __name__ == "__main__":
    main()