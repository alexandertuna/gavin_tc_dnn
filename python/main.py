"""
Main script to run Gavin's track embedding model training and visualization
"""
import argparse

from preprocess import Preprocessor
from train import Trainer
from viz import Plotter
from pathlib import Path


def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, default="/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiChargeXYZ.root",
                        help="Path to the input LSTNtuple ROOT file")
    parser.add_argument("--model", type=str, default="model_weights.pth",
                        help="Path to save or load the model weights")
    parser.add_argument("--pdf", type=str, default="plots.pdf",
                        help="Path to save the output plots in PDF format")
    parser.add_argument("--parallelism_test", action="store_true",
                        help="Run a test to check parallelism in data processing")
    parser.add_argument("--speed_test", action="store_true",
                        help="Run a speed test for data processing")
    parser.add_argument("--load_model", action="store_true",
                        help="Flag to load the model; if not set, the model will be trained")
    parser.add_argument("--emb_dim", type=int, default=6,
                        help="Dimensionality of the embedding space")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs for training the model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--load_features", action="store_true",
                        help="Flag to load precomputed features instead of computing them from scratch")
    parser.add_argument("--load_pairs", action="store_true",
                        help="Flag to load precomputed pairs instead of computing them from scratch")
    parser.add_argument("--upweight_displaced", type=float, default=5.0,
                        help="Upweight factor for displaced features")
    parser.add_argument("--delta_r2_cut", type=float, default=0.02,
                        help="Delta R^2 cut for feature selection")
    parser.add_argument("--use_phi_projection", action="store_true",
                        help="Flag to use position-based phi for PLS features")
    parser.add_argument("--use_phi_plus_pi", action="store_true",
                        help="Flag to use (phi, phi+pi) for features instead of (cosphi, sinphi)")
    parser.add_argument("--use_pls_deltaphi", action="store_true",
                        help="Flag to include pls deltaphi in features")
    parser.add_argument("--features_t5", type=str, default="features_t5.pkl",
                        help="Path to the precomputed T5 features file")
    parser.add_argument("--features_pls", type=str, default="features_pls.pkl",
                        help="Path to the precomputed PLS features file")
    parser.add_argument("--pairs_t5t5", type=str, default="pairs_t5t5.pkl",
                        help="Path to the precomputed T5-T5 pairs file")
    parser.add_argument("--pairs_t5pls", type=str, default="pairs_t5pls.pkl",
                        help="Path to the precomputed T5-PLS pairs file")
    parser.add_argument("--pairs_plspls", type=str, default="pairs_plspls.pkl",
                        help="Path to the precomputed PLS-PLS pairs file")
    return parser.parse_args()


def main():

    # Command line arguments
    args = options()
    file_path = Path(args.input)
    model_path = Path(args.model)
    pdf_path = Path(args.pdf)

    # Data processing
    processor = Preprocessor(file_path,
                             args.load_features,
                             args.load_pairs,
                             args.features_t5,
                             args.features_pls,
                             args.pairs_t5t5,
                             args.pairs_t5pls,
                             args.pairs_plspls,
                             use_phi_projection=args.use_phi_projection,
                             use_phi_plus_pi=args.use_phi_plus_pi,
                             use_pls_deltaphi=args.use_pls_deltaphi,
                             upweight_displaced=args.upweight_displaced,
                             delta_r2_cut=args.delta_r2_cut
                             )

    # Tests?
    if args.parallelism_test:
        processor.parallelism_test()
    if args.speed_test:
        processor.speed_test()

    # ML training
    trainer = Trainer(seed=args.seed,
                      emb_dim=args.emb_dim,
                      use_pls_deltaphi=args.use_pls_deltaphi,
                      # --------------------
                      bonus_features=processor.bonus_features,
                      # --------------------
                      X_left_train=processor.X_left_train,
                      X_left_test=processor.X_left_test,
                      X_right_train=processor.X_right_train,
                      X_right_test=processor.X_right_test,
                      y_t5_train=processor.y_t5_train,
                      y_t5_test=processor.y_t5_test,
                      w_t5_train=processor.w_t5_train,
                      w_t5_test=processor.w_t5_test,
                      # --------------------
                      X_pls_train=processor.X_pls_train,
                      X_pls_test=processor.X_pls_test,
                      X_t5raw_train=processor.X_t5raw_train,
                      X_t5raw_test=processor.X_t5raw_test,
                      y_pls_train=processor.y_pls_train,
                      y_pls_test=processor.y_pls_test,
                      w_pls_train=processor.w_pls_train,
                      w_pls_test=processor.w_pls_test,
                      )

    if not args.load_model:
        trainer.train(num_epochs=args.num_epochs)
        # trainer.print_thresholds()
        # trainer.print_weights_biases()
        trainer.save(model_path)
    else:
        trainer.load(model_path)

    plotter = Plotter(trainer)
    plotter.plot(pdf_path)

    # inspector = InspectT5T5(processor, trainer)
    # inspector.inspect()

if __name__ == "__main__":
    main()
