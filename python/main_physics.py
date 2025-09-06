"""
Main script to run Gavin's track embedding model training and visualization
"""
import argparse

from preprocess import PreprocessorPtEtaPhi
from train import Trainer
from viz import Plotter
from pathlib import Path


def options():
    default_dir = "/ceph/users/atuna/work/gavin_tc_dnn/experiments/embed_ptetaphi"
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--features_t5", type=str, default=f"{default_dir}/features_t5.pkl",
                        help="Path to the precomputed T5 features file")
    parser.add_argument("--features_pls", type=str, default=f"{default_dir}/features_pls.pkl",
                        help="Path to the precomputed PLS features file")
    parser.add_argument("--sim_features_t5", type=str, default=f"{default_dir}/sim_features_t5.pkl",
                        help="Path to the precomputed T5 sim features file")
    parser.add_argument("--sim_features_pls", type=str, default=f"{default_dir}/sim_features_pls.pkl",
                        help="Path to the precomputed PLS sim features file")
    parser.add_argument("--model", type=str, default="model_weights.pth",
                        help="Path to save or load the model weights")
    parser.add_argument("--pdf", type=str, default="plots.pdf",
                        help="Path to save the output plots in PDF format")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs for training the model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--upweight_displaced", type=float, default=5.0,
                        help="Upweight factor for displaced features")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train-test split")
    parser.add_argument("--recalculate_std", action="store_true",
                        help="Recalculate standard deviation from the data instead of using predefined values")
    return parser.parse_args()


def main():

    # Command line arguments
    args = options()
    model_path = Path(args.model)
    pdf_path = Path(args.pdf)

    # Data processing
    processor = PreprocessorPtEtaPhi(args.features_t5,
                                     args.features_pls,
                                     args.sim_features_t5,
                                     args.sim_features_pls,
                                     test_size=args.test_size,
                                     recalculate_std=args.recalculate_std,
                                     )

    return

    # ML training
    trainer = Trainer(seed=args.seed,
                      num_epochs=args.num_epochs,
                      emb_dim=args.emb_dim,
                      use_pls_deltaphi=args.use_pls_deltaphi,
                      use_scheduler=args.use_scheduler,
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
        trainer.train()
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
