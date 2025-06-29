import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

from ml import SiameseDataset, PLST5Dataset, EmbeddingNetT5, EmbeddingNetpLS, ContrastiveLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

class Trainer:

    def __init__(self,
                 X_left_train,
                 X_left_test,
                 X_right_train,
                 X_right_test,
                 y_t5_train,
                 y_t5_test,
                 w_t5_train,
                 w_t5_test,
                 # ------------
                 X_pls_train,
                 X_pls_test,
                 X_t5raw_train,
                 X_t5raw_test,
                 y_pls_train,
                 y_pls_test,
                 w_pls_train,
                 w_pls_test,
                 ):

        self.X_left_test = X_left_test
        self.X_right_test = X_right_test

        print("Creating datasets ...")
        train_t5_ds = SiameseDataset(X_left_train, X_right_train, y_t5_train, w_t5_train)
        test_t5_ds  = SiameseDataset(X_left_test,  X_right_test,  y_t5_test,  w_t5_test)

        train_pls_ds = PLST5Dataset(X_pls_train, X_t5raw_train, y_pls_train, w_pls_train)
        test_pls_ds  = PLST5Dataset(X_pls_test,  X_t5raw_test,  y_pls_test,  w_pls_test)

        batch_size = 1024
        num_workers = min(os.cpu_count() or 4, 8)

        print("Creating loaders ...")
        self.train_t5_loader = DataLoader(train_t5_ds, batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=True)
        self.test_t5_loader  = DataLoader(test_t5_ds,  batch_size, shuffle=False,
                                          num_workers=num_workers, pin_memory=True)
        self.train_pls_loader = DataLoader(train_pls_ds, batch_size, shuffle=True,
                                           num_workers=num_workers, pin_memory=True)
        self.test_pls_loader  = DataLoader(test_pls_ds,  batch_size, shuffle=False,
                                           num_workers=num_workers, pin_memory=True)

        print("Loaders ready:",
            f"T5 train {len(train_t5_ds)}, pLS-T5 train {len(train_pls_ds)}")

        # contrastive loss (reuse)
        self.criterion = ContrastiveLoss(margin=1.0)

        # instantiate and send to GPU/CPU
        print("Creating embedding networks ...")
        self.embed_t5 = EmbeddingNetT5().to(DEVICE)
        self.embed_pls = EmbeddingNetpLS().to(DEVICE)

        # joint optimizer over both nets
        print("Creating optimizer ...")
        self.optimizer = optim.Adam(
            list(self.embed_t5.parameters()) + list(self.embed_pls.parameters()),
            lr=0.0025
        )


    def train(self):
        num_epochs = 200
        print(time.strftime("Time: %Y-%m-%d %H:%M:%S", time.localtime()))

        for epoch in range(1, num_epochs+1):
            self.embed_t5.train(); self.embed_pls.train()
            total_loss = 0.0
            total_t5   = 0.0
            total_pls  = 0.0

            # zip will stop at the shorter loader; you can also use itertools.cycle if needed
            for (l, r, y0, w0), (p5, t5f, y1, w1) in zip(self.train_t5_loader, self.train_pls_loader):
                # to device
                l   = l.to(DEVICE);   r    = r.to(DEVICE)
                y0_ = y0.to(DEVICE);  w0_  = w0.to(DEVICE)
                p5  = p5.to(DEVICE);  t5f_ = t5f.to(DEVICE)
                y1_ = y1.to(DEVICE);  w1_  = w1.to(DEVICE)

                # --- T5–T5 forward & loss ---
                e_l = self.embed_t5(l);  e_r = self.embed_t5(r)
                d0 = torch.sqrt(((e_l-e_r)**2).sum(1,keepdim=True) + 1e-6)
                loss0 = self.criterion(d0, y0_, w0_)

                # --- pLS-T5 forward & loss ---
                e_p5 = self.embed_pls(p5)
                e_t5 = self.embed_t5(t5f_)
                d1 = torch.sqrt(((e_p5-e_t5)**2).sum(1,keepdim=True) + 1e-6)
                loss1 = self.criterion(d1, y1_, w1_)

                loss = loss0 + loss1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_t5  += loss0.item()
                total_pls += loss1.item()

            avg_loss   = total_loss / len(self.train_pls_loader)
            avg_t5     = total_t5   / len(self.train_t5_loader)
            avg_pls    = total_pls  / len(self.train_pls_loader)
            print(f"Epoch {epoch}/{num_epochs}:  JointLoss={avg_loss:.4f}  "
                f"T5={avg_t5:.4f}  pLS={avg_pls:.4f}")

        # disable training mode
        self.embed_pls.eval(); self.embed_t5.eval()


    def print_weights_biases(self):
        print("*"*50)
        print(time.strftime("Time: %Y-%m-%d %H:%M:%S", time.localtime()))
        print("T5 embedding model weights and biases:")
        print_model_weights_biases(self.embed_t5)
        print("*"*50)

        print("*"*50)
        print("PLS embedding model weights and biases:")
        print_model_weights_biases(self.embed_pls)
        print("*"*50)


    def save(self, path):
        print(f"Saving model to {path}")
        torch.save({
            'embed_t5': self.embed_t5.state_dict(),
            'embed_pls': self.embed_pls.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)


    def load(self, path):
        print(f"Loading model from {path}")
        checkpoint = torch.load(path) # map_location=DEVICE
        self.embed_t5.load_state_dict(checkpoint['embed_t5'])
        self.embed_pls.load_state_dict(checkpoint['embed_pls'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.embed_t5.to(DEVICE)
        self.embed_pls.to(DEVICE)


def print_formatted_weights_biases(weights, biases, layer_name):
    # Print biases
    print(f"HOST_DEVICE_CONSTANT float bias_{layer_name}[{len(biases)}] = {{")
    print(", ".join(f"{b:.7f}f" for b in biases) + " };")
    print()

    # Print weights
    print(f"HOST_DEVICE_CONSTANT const float wgtT_{layer_name}[{len(weights[0])}][{len(weights)}] = {{")
    for row in weights.T:
        formatted_row = ", ".join(f"{w:.7f}f" for w in row)
        print(f"{{ {formatted_row} }},")
    print("};")
    print()


def print_model_weights_biases(model):
    # Make sure the model is in evaluation mode
    model.eval()

    # Iterate through all named modules in the model
    for name, module in model.named_modules():
        # Check if the module is a linear layer
        if isinstance(module, nn.Linear):
            # Get weights and biases
            weights = module.weight.data.cpu().numpy()
            biases = module.bias.data.cpu().numpy()

            # Print formatted weights and biases
            print_formatted_weights_biases(weights, biases, name.replace('.', '_'))