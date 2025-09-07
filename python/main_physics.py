"""
Main script to run Gavin's track embedding model training and visualization
"""
import argparse

from preprocess import Preprocessor, PreprocessorPtEtaPhi
from train import Trainer, TrainerPtEtaPhi
from viz import PlotterPtEtaPhi
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
    parser.add_argument("--model", type=str, default="model_weights_ptetaphi.pth",
                        help="Path to save or load the model weights")
    parser.add_argument("--embedding_model", type=str, default="model_weights.pth",
                        help="Path to save or load the Gavin embedding model weights")
    parser.add_argument("--pdf", type=str, default="plots_ptetaphi.pdf",
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
    parser.add_argument("--load_model", action="store_true",
                        help="Load model weights from the specified path instead of training")
    parser.add_argument("--load_embedding_model", action="store_true",
                        help="Load Gavin embedding model weights from the specified path instead of training")
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

    if not args.load_model:
        trainer.train()
        # trainer.print_thresholds()
        # trainer.print_weights_biases()
        trainer.save(model_path)
    else:
        trainer.load(model_path)

    # Load default embedding model for ROC curve comparison
    train_emb = None
    if args.load_embedding_model:
        print(f"Loading Gavin embedding model weights from {args.embedding_model}")
        proc_emb = Preprocessor(root_path=None,
                                LOAD_FEATURES=True,
                                LOAD_PAIRS=True,
                                FEATURES_T5="features_t5.pkl",
                                FEATURES_PLS="features_pls.pkl",
                                PAIRS_T5T5="pairs_t5t5.pkl",
                                PAIRS_T5PLS="pairs_t5pls.pkl",
                                PAIRS_PLSPLS="pairs_plspls.pkl",
                                use_phi_projection=False,
                                use_phi_plus_pi=False,
                                use_pls_deltaphi=False,
                                use_no_phi=False,
                                upweight_displaced=5.0,
                                delta_r2_cut=0.02,
                                test_size=0.2,
                                dont_swap_lr=False,
                                )
        train_emb = Trainer(seed=None,
                            num_epochs=None,
                            emb_dim=6,
                            use_pls_deltaphi=None,
                            use_scheduler=args.use_scheduler,
                            # --------------------
                            bonus_features=proc_emb.bonus_features,
                            # --------------------
                            X_left_train=proc_emb.X_left_train,
                            X_left_test=proc_emb.X_left_test,
                            X_right_train=proc_emb.X_right_train,
                            X_right_test=proc_emb.X_right_test,
                            y_t5_train=proc_emb.y_t5_train,
                            y_t5_test=proc_emb.y_t5_test,
                            w_t5_train=proc_emb.w_t5_train,
                            w_t5_test=proc_emb.w_t5_test,
                            # --------------------
                            X_pls_train=proc_emb.X_pls_train,
                            X_pls_test=proc_emb.X_pls_test,
                            X_t5raw_train=proc_emb.X_t5raw_train,
                            X_t5raw_test=proc_emb.X_t5raw_test,
                            y_pls_train=proc_emb.y_pls_train,
                            y_pls_test=proc_emb.y_pls_test,
                            w_pls_train=proc_emb.w_pls_train,
                            w_pls_test=proc_emb.w_pls_test,
                            )
        train_emb.load(args.embedding_model)


    plotter = PlotterPtEtaPhi(trainer, train_emb)
    plotter.plot(pdf_path)

    # inspector = InspectT5T5(processor, trainer)
    # inspector.inspect()

if __name__ == "__main__":
    main()
