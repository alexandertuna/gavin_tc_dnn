"""
Main script to run Gavin's track embedding model training and visualization
"""
import argparse

from preprocess import PreprocessorPtEtaPhi
from train import TrainerPtEtaPhi
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
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Use learning rate scheduler during training")
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


    # ML training
    trainer = TrainerPtEtaPhi(seed=args.seed,
                              num_epochs=args.num_epochs,
                              use_scheduler=args.use_scheduler,
                              # --------------------
                              bonus_features=processor.bonus_features,
                                # --------------------
                              features_t5_train=processor.features_t5_train,
                              features_t5_test=processor.features_t5_test,
                              features_pls_train=processor.features_pls_train,
                              features_pls_test=processor.features_pls_test,
                              sim_features_t5_train=processor.sim_features_t5_train,
                              sim_features_t5_test=processor.sim_features_t5_test,
                              sim_features_pls_train=processor.sim_features_pls_train,
                              sim_features_pls_test=processor.sim_features_pls_test,
                              )
    
    return 

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
