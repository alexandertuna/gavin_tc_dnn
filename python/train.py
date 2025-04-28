#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import pickle

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# local imports
import preprocess

def main():
    # train()
    plot_stuff()


def train():

    file_path = "../data/LSTNtuple.root"
    branches, features, sim_indices, X_left, X_right, y = preprocess.preprocess_data(file_path)

    pdf_path = "plots.pdf"
    pdf = PdfPages(pdf_path)

    # Split into training and testing sets.
    X_left_train, X_left_test, X_right_train, X_right_test, y_train, y_test = train_test_split(
        X_left, X_right, y, test_size=0.2, random_state=42
    )

    batch_size = 128
    train_dataset = SiameseDataset(X_left_train, X_right_train, y_train)
    test_dataset = SiameseDataset(X_left_test, X_right_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = 4       # four input features
    embedding_dim = 4   # embedding size

    embedding_net = EmbeddingNet(input_dim, embedding_dim)
    model = SiameseNet(embedding_net)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ------------------------------------------------------------
    # 5. Train the Model
    # ------------------------------------------------------------
    num_epochs = 30 # 300
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_left, batch_right, batch_label in train_loader:
            optimizer.zero_grad()
            distances = model(batch_left, batch_right)
            loss = criterion(distances, batch_label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_left.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation step.
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_left, batch_right, batch_label in test_loader:
                distances = model(batch_left, batch_right)
                loss = criterion(distances, batch_label)
                running_val_loss += loss.item() * batch_left.size(0)
        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - now {time.strftime("%Y_%m_%d_%Hh%Mm%Ss")}")

    # ------------------------------------------------------------
    # 6. Performance Evaluation and Plotting
    # ------------------------------------------------------------
    model.eval()
    all_distances = []
    all_labels = []
    with torch.no_grad():
        for batch_left, batch_right, batch_label in test_loader:
            distances = model(batch_left, batch_right)
            all_distances.append(distances.cpu().numpy())
            all_labels.append(batch_label.cpu().numpy())

    all_distances = np.concatenate(all_distances).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    print("Test distances range: ", all_distances.min(), all_distances.max())

    # For a range of thresholds, count duplicate pairs (true positives) and non-duplicate pairs (false positives)
    thresholds = np.linspace(all_distances.min(), all_distances.max(), 100)
    tp_list = []  # duplicate pairs correctly identified (true positives)
    fp_list = []  # non-duplicate pairs incorrectly flagged (false positives)

    for t in thresholds:
        pred_duplicate = all_distances < t  # predict duplicate if distance < threshold
        tp = np.sum((pred_duplicate == True) & (all_labels == 0))
        fp = np.sum((pred_duplicate == True) & (all_labels == 1))
        tp_list.append(tp)
        fp_list.append(fp)

    # Plot the performance curve.
    plt.figure(figsize=(8, 6))
    plt.plot(fp_list, tp_list, marker='o')
    plt.xlabel("Non-Duplicate Pairs (False Positives) Removed")
    plt.ylabel("Duplicate Pairs (True Positives) Removed")
    plt.title("Duplicate Removal Performance")
    plt.grid(True)
    pdf.savefig()
    plt.close()
    # plt.show()

    # ----------------------------
    # 6. Performance Evaluation and Additional Plots
    # ----------------------------
    model.eval()
    all_distances = []
    all_labels = []
    with torch.no_grad():
        for batch_left, batch_right, batch_label in test_loader:
            distances = model(batch_left, batch_right)
            all_distances.append(distances.cpu().numpy())
            all_labels.append(batch_label.cpu().numpy())

    all_distances = np.concatenate(all_distances).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    print("Test distances range: ", all_distances.min(), all_distances.max())

    # ----------------------------
    # Plot 1: Performance at 99.5% Non-Duplicate Retention
    # ----------------------------
    # Separate distances by label.
    nondup_distances = all_distances[all_labels==1]  # non-duplicate pairs
    dup_distances = all_distances[all_labels==0]      # duplicate pairs

    # Determine threshold such that 99.5% of non-duplicate pairs are retained (i.e. only 0.5% fall below the threshold).
    threshold_nondup = np.percentile(nondup_distances, 0.0001)
    print("Threshold for 99.99% retention of non-duplicates:", threshold_nondup)

    # Count how many duplicate pairs would be rejected (i.e. have distance below the threshold).
    tp_at_threshold = np.sum(dup_distances < threshold_nondup)
    total_dup = len(dup_distances)
    print("Duplicate pairs rejected (distance < threshold):", tp_at_threshold, "out of", total_dup)

    plt.figure(figsize=(8,6))
    plt.hist(nondup_distances, bins=50, alpha=0.5, label="Non-Duplicate Pairs")
    plt.hist(dup_distances, bins=50, alpha=0.5, label="Duplicate Pairs")
    plt.axvline(x=threshold_nondup, color='red', linestyle='--', 
                label=f"Threshold (99.99% non-dup kept)\n{threshold_nondup:.3f}")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.title("Distribution of Pair Distances")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    pdf.savefig()
    plt.close()
    # plt.show()


    # ----------------------------
    # Plot 2: Embedding Space Visualization
    # ----------------------------
    # Choose one event (e.g., event 0) to visualize its tracks in the embedding space.
    event_idx = 0  # you can change this to any event you like
    event_features = torch.tensor(features[event_idx].astype(np.float32))
    with torch.no_grad():
        event_embeddings = embedding_net(event_features).cpu().numpy()

    # Get the corresponding sim indices for this event.
    event_sim = sim_indices[event_idx]
    # For visualization, if sim index is None, assign a fixed value (-1).
    colors = np.array([sim if sim is not None else -1 for sim in event_sim], dtype=float)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(event_embeddings[:, 0], event_embeddings[:, 1], c=colors, cmap='viridis', s=50)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.title(f"Embedding Space for Event {event_idx}")
    plt.colorbar(sc, label="Sim Index")
    pdf.savefig()
    plt.close()
    # plt.show()

    # Plot the training and validation loss history.
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()
    # plt.show()


    # ----------------------------
    # Choose an event index to visualize
    # ----------------------------
    event_idx = 0  # Change to the event you want to plot

    # Convert the features for this event to a PyTorch tensor.
    event_features = torch.tensor(features[event_idx].astype(np.float32))

    # Forward pass through the embedding network to get 4D embeddings (shape: [n_tracks, 4])
    with torch.no_grad():
        event_embeddings = embedding_net(event_features).cpu().numpy()

    # Retrieve track types for this event (shape: [n_tracks])
    event_types = branches['tc_type'][event_idx]

    # We'll plot these dimension pairs from the 4D embedding space.
    dim_pairs = [(0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3), (2, 3)]

    # Create a figure with subplots for each pair
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs = axs.flatten()  # Flatten so we can index easily

    # Define a dictionary that assigns each tc_type a distinct color.
    type_color_map = {
        4: 'red',
        5: 'blue',
        7: 'green',
        8: 'purple'
    }

    # Define a dictionary that maps each tc_type to a more descriptive label.
    type_label_map = {
        4: 'pT5',
        5: 'pT3',
        7: 'T5',
        8: 'pLS'
    }

    for idx, (x_dim, y_dim) in enumerate(dim_pairs):
        ax = axs[idx]

        # Plot each point according to its tc_type
        for i, ttype in enumerate(event_types):
            # Look up the color and label. If unknown type, fallback to "gray" / "unknown".
            color = type_color_map.get(ttype, 'gray')
            label = type_label_map.get(ttype, 'unknown')

            ax.scatter(
                event_embeddings[i, x_dim],
                event_embeddings[i, y_dim],
                color=color,
                s=40,
                alpha=0.7,
                edgecolors='k',
                label=label  # We'll handle duplicates in the legend below
            )

        ax.set_xlabel(f"Embedding Dim {x_dim}")
        ax.set_ylabel(f"Embedding Dim {y_dim}")
        ax.set_title(f"Dims {x_dim} vs {y_dim}")

        # Collect legend entries for this subplot, dropping duplicates
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))  # label -> handle
        ax.legend(unique.values(), unique.keys(), loc='best')

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    # plt.show()

    # ----------------------------
    # Choose an event index to visualize
    # ----------------------------
    event_idx = 0  # Change to the event you want to plot

    # Convert the features for this event to a PyTorch tensor and embed them.
    event_features = torch.tensor(features[event_idx].astype(np.float32))
    with torch.no_grad():
        event_embeddings = embedding_net(event_features).cpu().numpy()  # shape: [n_tracks, 4]

    # Retrieve the duplicate flags for this event (0 or 1 per track)
    event_isdup = branches['tc_isDuplicate'][event_idx]  # shape: [n_tracks]

    # We'll plot these dimension pairs from the 4D embedding space.
    dim_pairs = [(0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3), (2, 3)]

    # Create a figure with subplots for each pair
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs = axs.flatten()  # Flatten so we can index easily

    for idx, (x_dim, y_dim) in enumerate(dim_pairs):
        ax = axs[idx]

        # Create boolean masks for duplicates vs. non-duplicates
        mask_dup = (event_isdup == 1)
        mask_nondup = (event_isdup == 0)

        # Plot non-duplicate tracks in blue
        ax.scatter(
            event_embeddings[mask_nondup, x_dim],
            event_embeddings[mask_nondup, y_dim],
            color='blue',
            s=40,
            alpha=0.7,
            edgecolors='k',
            label='Non-Duplicate Tracks'
        )

        # Plot duplicate tracks in red
        ax.scatter(
            event_embeddings[mask_dup, x_dim],
            event_embeddings[mask_dup, y_dim],
            color='red',
            s=40,
            alpha=0.7,
            edgecolors='k',
            label='Duplicate Tracks'
        )

        ax.set_xlabel(f"Embedding Dim {x_dim}")
        ax.set_ylabel(f"Embedding Dim {y_dim}")
        ax.set_title(f"Dims {x_dim} vs {y_dim}")

        # Add a legend to each subplot, removing duplicate entries
        handles, labels = ax.get_legend_handles_labels()
        # Convert to a dict to drop duplicates
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='best')

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    # plt.show()

    num_events = 3

    # Initialize accumulators.
    sum_removed_nondup_real = 0
    sum_total_real = 0

    sum_removed_duplicates = 0
    sum_total_duplicates = 0

    for event_idx in range(num_events):
        # Get features and compute embeddings for event_idx
        event_features = torch.tensor(features[event_idx].astype(np.float32))
        with torch.no_grad():
            event_embeddings = embedding_net(event_features).cpu().numpy()

        # Get the duplicate flag and sim indices for each track in the event.
        event_isdup = branches['tc_isDuplicate'][event_idx]  # Boolean or 0/1
        event_sim = sim_indices[event_idx]  # None => fake track, else real track index

        n_tracks = event_embeddings.shape[0]

        # Count total real and fake tracks in this event.
        total_real = sum(1 for s in event_sim if s is not None)
        total_fake = n_tracks - total_real

        # Count how many are flagged as duplicates vs. non-duplicates before removal.
        total_duplicates_before = sum(1 for flag in event_isdup if flag)
        total_nonduplicates_before = n_tracks - total_duplicates_before

        print(f"Event {event_idx}:")
        print("Before duplicate removal:")
        print(f"  Total tracks: {n_tracks}")
        print(f"  Real tracks: {total_real}")
        print(f"  Fake tracks: {total_fake}")
        print(f"  Duplicate tracks: {total_duplicates_before}")
        print(f"  Non-duplicate tracks: {total_nonduplicates_before}")

        # --- Duplicate Removal: Greedy Algorithm ---
        threshold = 0.05
        keep_mask = np.ones(n_tracks, dtype=bool)

        for i in range(n_tracks):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n_tracks):
                if not keep_mask[j]:
                    continue
                d = np.linalg.norm(event_embeddings[i] - event_embeddings[j])
                if d < threshold:
                    keep_mask[j] = False

        after_count = np.sum(keep_mask)
        removed_count = n_tracks - after_count

        # Post-removal statistics
        removed_indices = np.where(~keep_mask)[0]
        removed_duplicates = sum(1 for i in removed_indices if event_isdup[i])
        removed_nonduplicates = removed_count - removed_duplicates

        # Among removed non-duplicates, count real vs. fake.
        removed_nondup_real = sum(
            1 for i in removed_indices if (not event_isdup[i] and event_sim[i] is not None)
        )
        removed_nondup_fake = sum(
            1 for i in removed_indices if (not event_isdup[i] and event_sim[i] is None)
        )

        print("After duplicate removal:")
        print(f"  Total tracks kept: {after_count}")
        print(f"  Total tracks removed: {removed_count}")
        print(f"  Removed tracks flagged as duplicates: {removed_duplicates}")
        print(f"  Removed tracks flagged as non-duplicates: {removed_nonduplicates}")
        print(f"     - Of these non-duplicates, real tracks removed: {removed_nondup_real}")
        print(f"     - Of these non-duplicates, fake tracks removed: {removed_nondup_fake}")
        print("----------")

        # Accumulate for overall statistics
        sum_removed_nondup_real += removed_nondup_real
        sum_total_real += total_real

        sum_removed_duplicates += removed_duplicates
        sum_total_duplicates += total_duplicates_before

    # ---- After processing all events, compute overall percentages. ----

    if sum_total_real > 0:
        percent_real_removed = 100.0 * sum_removed_nondup_real / sum_total_real
    else:
        percent_real_removed = 0.0

    if sum_total_duplicates > 0:
        percent_duplicates_removed = 100.0 * sum_removed_duplicates / sum_total_duplicates
    else:
        percent_duplicates_removed = 0.0

    print("=== Overall Summary (across all events) ===")
    print(f"Percent of real tracks removed (non-duplicate real tracks): {percent_real_removed:.2f}%")
    print(f"Percent of duplicates removed: {percent_duplicates_removed:.2f}%")

    print_model_weights_biases(model)

    # save relevant things to file
    with open("model_data.pkl", "wb") as fi:
        pickle.dump({
            "features": features,
            "embedding_net": embedding_net,
            "test_loader": test_loader,
            "model": model,
        }, fi)

    pdf.close()



def plot_stuff():

    # Load the data back
    with open("model_data.pkl", "rb") as fi:
        di = pickle.load(fi)

    pdf_path = "plots.pdf"
    pdf = PdfPages(pdf_path)

    # alex plots
    # print("features:", type(features))
    # print("embedding_net:", type(embedding_net))
    # print("test_loader:", type(test_loader))
    # print("model:", type(model))
    plot_inputs_vs_embeddings(di["features"], di["embedding_net"], pdf)
    plot_derived_vs_embedding(di["test_loader"], di["model"], pdf)

    # Save the PDF file
    pdf.close()


# ------------------------------------------------------------
# 3. Create a PyTorch Dataset and DataLoader
# ------------------------------------------------------------
class SiameseDataset(Dataset):
    def __init__(self, X_left, X_right, y):
        self.X_left = X_left.astype(np.float32)
        self.X_right = X_right.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_left[idx], self.X_right[idx], self.y[idx]


# ------------------------------------------------------------
# 4. Define the Siamese Network in PyTorch
# ------------------------------------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim=4, embedding_dim=2):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)  # New hidden layer with 16 units
        self.fc3 = nn.Linear(16, embedding_dim)  # Final output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        # Compute Euclidean distance between the two embeddings.
        distance = torch.sqrt(torch.sum((out1 - out2)**2, dim=1, keepdim=True) + 1e-6)
        return distance

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # For duplicate pairs (label 0): loss = distance^2.
        # For non-duplicates (label 1): loss = max(margin - distance, 0)^2.
        loss_similar = (1 - label) * torch.pow(distance, 2)
        loss_dissimilar = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss




def print_formatted_weights_biases(weights, biases, layer_name):
    # Print biases
    print(f"HOST_DEVICE_CONSTANT float bias_{layer_name}[{len(biases)}] = {{")
    print(", ".join(f"{b:.7f}f" for b in biases) + " };")
    print()

    # Print weights
    print(f"HOST_DEVICE_CONSTANT float wgtT_{layer_name}[{len(weights[0])}][{len(weights)}] = {{")
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


def plot_inputs_vs_embeddings(features, embedding_net, pdf):
    n_inp = 4
    n_emb = 4
    
    # event_idx = 0
    # event_features = torch.tensor(features[event_idx].astype(np.float32))
    print("len(features)", len(features))
    event_idxs = slice(0, 175)
    event_features = torch.tensor(np.concat(features[event_idxs]).astype(np.float32))
    with torch.no_grad():
        event_embeddings = embedding_net(event_features).cpu().numpy()
    print(event_features.shape)
    print(event_embeddings.shape)

    titles_inp = [
        "log10(pT)",
        "eta (scaled)",
        "phi (scaled)",
        "type (scaled)",
    ]
    bins_inp = [
        np.arange(-0.1, 2.1, 0.05),
        np.arange(-1.1, 1.2, 0.05),
        np.arange(-1.1, 1.2, 0.05),
        np.arange(-1.1, 1.2, 0.2),
    ]
    bins_emb = [
        np.arange(-5, 15, 0.1),
        np.arange(-2, 5, 0.1),
        np.arange(-11, 6, 0.1),
        np.arange(-1, 3, 0.05),
    ]
    
    fig, ax = plt.subplots(ncols=n_inp, figsize=(14, 4))
    for idx in range(n_inp):
        ax[idx].hist(event_features[:, idx], bins=bins_inp[idx])
        ax[idx].set_xlabel(titles_inp[idx])
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(ncols=n_emb, figsize=(14, 4))
    for idx in range(n_emb):
        ax[idx].hist(event_embeddings[:, idx], bins=bins_emb[idx])
        ax[idx].set_xlabel(f"Embedding dimension {idx}")
    pdf.savefig()
    plt.close()

    for inp in range(n_inp):
        fig, ax = plt.subplots(ncols=n_emb, figsize=(14, 4))
        for emb in range(n_emb):
            _, _, _, im = ax[emb].hist2d(event_features[:, inp],
                                         event_embeddings[:, emb],
                                         bins=[bins_inp[inp], bins_emb[emb]],
                                         cmin=0.5,
                                         )
            ax[emb].set_xlabel(titles_inp[inp])
            ax[emb].set_ylabel(f"Embedding dimension {emb}")
        pdf.savefig()
        plt.close()


def dangle(x, y):
    return np.min([(2 * np.pi) - np.abs(x - y), np.abs(x - y)], axis=0)

def plot_derived_vs_embedding(test_loader, model, pdf):

    distances, labels, dpts, detas, dphis, drs = [], [], [], [], [], []
    with torch.no_grad():
        for batch_left, batch_right, batch_label in test_loader:
            ds = model(batch_left, batch_right)
            pt_l = (10 ** batch_left[:, 0]).cpu().numpy()
            pt_r = (10 ** batch_right[:, 0]).cpu().numpy()
            eta_l = (4 * batch_left[:, 1]).cpu().numpy()
            eta_r = (4 * batch_right[:, 1]).cpu().numpy()
            phi_l = (3.1415926 * batch_left[:, 2]).cpu().numpy()
            phi_r = (3.1415926 * batch_right[:, 2]).cpu().numpy()
            # print(batch_left.shape)
            dpts.append(pt_l - pt_r)
            detas.append(eta_l - eta_r)
            dphis.append(dangle(phi_l, phi_r))
            drs.append( ((eta_l - eta_r)**2 + (dangle(phi_l, phi_r))**2) ** 0.5 )
            labels.append(batch_label.cpu().numpy())
            distances.append(ds)
            # break

    def flatten(li):
        return np.concatenate(li).flatten()

    distances = flatten(distances)
    labels = np.abs(flatten(labels))
    dpts = np.abs(flatten(dpts))
    detas = np.abs(flatten(detas))
    dphis = np.abs(flatten(dphis))
    drs = np.abs(flatten(drs))

    masks = [
        labels == 0,
        labels == 1,
    ]
    bins_distance = np.arange(0, 20, 0.1)
    bins_dpt = np.arange(-1, 5, 0.1)
    bins_deta = np.arange(0, 7, 0.1)
    bins_dphi = np.arange(0, 3.25, 0.05)
    bins_dr = np.arange(0, 7, 0.1)
    
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))
    for idx, mask in enumerate(masks):
        ax[idx, 0].hist(dpts[mask], bins=bins_dpt)
        ax[idx, 1].hist(detas[mask], bins=bins_deta)
        ax[idx, 2].hist(dphis[mask], bins=bins_dphi)
        ax[idx, 3].hist(drs[mask], bins=bins_dr)
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))
    for idx, mask in enumerate(masks):
        ax[idx, 0].hist2d(dpts[mask], distances[mask], cmin=0.5, bins=[bins_dpt, bins_distance])
        ax[idx, 1].hist2d(detas[mask], distances[mask], cmin=0.5, bins=[bins_deta, bins_distance])
        ax[idx, 2].hist2d(dphis[mask], distances[mask], cmin=0.5, bins=[bins_dphi, bins_distance])
        ax[idx, 3].hist2d(drs[mask], distances[mask], cmin=0.5, bins=[bins_dr, bins_distance])
    pdf.savefig()
    plt.close()


if __name__ == "__main__":
    main()
