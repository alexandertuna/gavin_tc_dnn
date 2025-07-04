from preprocess import Preprocessor
from train import Trainer
from viz import Plotter
from inspect_t5t5 import InspectT5T5
from pathlib import Path

def main():
    model_path = Path("model_weights.pth")
    pdf_path = Path("plots.pdf")
    file_path = "/ceph/users/atuna/work/gavin_tc_dnn/python/pls_t5_embed.root"
    # file_path = "/Users/alexandertuna/Downloads/cms/gavin_tc_dnn/data/pls_t5_embed.root"
    processor = Preprocessor(file_path)
    # processor.speed_test()
    # trainer = Trainer(processor.X_left_train,
    #                   processor.X_left_test,
    #                   processor.X_right_train,
    #                   processor.X_right_test,
    #                   processor.y_t5_train,
    #                   processor.y_t5_test,
    #                   processor.w_t5_train,
    #                   processor.w_t5_test,
    #                   # --------------------
    #                   processor.X_pls_train,
    #                   processor.X_pls_test,
    #                   processor.X_t5raw_train,
    #                   processor.X_t5raw_test,
    #                   processor.y_pls_train,
    #                   processor.y_pls_test,
    #                   processor.w_pls_train,
    #                   processor.w_pls_test,
    #                   )

    # if False:
    #     trainer.train()
    #     trainer.save(model_path)
    # else:
    #     trainer.load(model_path)

    # plotter = Plotter(trainer)
    # plotter.plot(pdf_path)

    # inspector = InspectT5T5(processor, trainer)
    # inspector.inspect()

if __name__ == "__main__":
    main()
