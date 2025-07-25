#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import pickle

plt.rcParams.update({'font.size': 16})

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# local imports
import preprocess
from ml import SiameseDataset, EmbeddingNet, SiameseNet, ContrastiveLoss

def main():

    file_path = "../data/LSTNtuple.root"
    # file_path = "../data/LSTNtuple.cutNA.root"
    branches, features, sim_indices, X_left, X_right, y = preprocess.preprocess_data(file_path)
    return 

    pdf_path = "train.pdf"
    pdf = PdfPages(pdf_path)

    # Split into training and testing sets.
    X_left_train, X_left_test, X_right_train, X_right_test, y_train, y_test = train_test_split(
        X_left, X_right, y, test_size=0.2, random_state=42
    )

    # quick plot of types of pairs
    dup_y = y_train == 0
    dup_n = y_train == 1
    types = {}
    type_l_train = X_left_train[:, 3] * 2 + 6 - 4
    type_r_train = X_right_train[:, 3] * 2 + 6 - 4
    types["l_train_dup_y"] = type_l_train[dup_y]
    types["l_train_dup_n"] = type_l_train[dup_n]
    types["r_train_dup_y"] = type_r_train[dup_y]
    types["r_train_dup_n"] = type_r_train[dup_n]

    for subset in [
        "train_dup_y",
        "train_dup_n",
    ]:
        fig, ax = plt.subplots(figsize=(8, 8))
        hist, _, _, im = ax.hist2d(types[f"l_{subset}"],
                                   types[f"r_{subset}"],
                                   # bins=(np.arange(0, 4, 0.1), np.arange(0, 4, 0.1)),
                                   cmin=0.5,
                                   cmap='gist_rainbow',
                                   )

        type_labels = ["T5", "PT3", "PT5", "PLS"]
        # type_types = [4, 5, 7, 8]
        type_types = [0, 1, 3, 4]
        ax.set_xticks(type_types, labels=type_labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(type_types, labels=type_labels)
        for i in type_types:
            for j in type_types:
                pass # ax.text(j, i, hist[j, i], ha="center", va="center", color="w")

        title_noun = "Duplicates" if "dup_y" in subset else "Non-duplicates"
        ax.set_xlabel("Left Track Type")
        ax.set_ylabel("Right Track Type")
        ax.set_title(f"Left vs Right Track Type ({title_noun})")
        ax.grid(True)

        fig.colorbar(im, ax=ax, label="Pairs")
        fig.subplots_adjust(left=0.15, right=0.90, top=0.95, bottom=0.13)
        pdf.savefig()
        plt.close()


    batch_size = 128
    train_dataset = SiameseDataset(X_left_train, X_right_train, y_train)
    test_dataset = SiameseDataset(X_left_test, X_right_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_dataset_deltaR = make_deltaR(X_left_test, X_right_test)

    input_dim = 4       # four input features
    embedding_dim = 4   # embedding size

    embedding_net = EmbeddingNet(input_dim, embedding_dim)
    model = SiameseNet(embedding_net)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ------------------------------------------------------------
    # 5. Train the Model
    # ------------------------------------------------------------
    num_epochs = 10 # 300
    train_losses = []
    val_losses = []
    train_losses_similar = []
    train_losses_dissimilar = []
    val_losses_similar = []
    val_losses_dissimilar = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_loss_similar = 0.0
        running_loss_dissimilar = 0.0
        for batch_left, batch_right, batch_label in train_loader:
            optimizer.zero_grad()
            distances = model(batch_left, batch_right)
            loss, loss_similar, loss_dissimilar = criterion(distances, batch_label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_left.size(0)
            running_loss_similar += loss_similar.item() * batch_left.size(0)
            running_loss_dissimilar += loss_dissimilar.item() * batch_left.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_losses_similar.append(running_loss_similar / len(train_loader.dataset))
        train_losses_dissimilar.append(running_loss_dissimilar / len(train_loader.dataset))


        # Validation step.
        model.eval()
        running_val_loss = 0.0
        running_val_loss_similar = 0.0
        running_val_loss_dissimilar = 0.0
        with torch.no_grad():
            for batch_left, batch_right, batch_label in test_loader:
                distances = model(batch_left, batch_right)
                loss, loss_similar, loss_dissimilar = criterion(distances, batch_label)
                running_val_loss += loss.item() * batch_left.size(0)
                running_val_loss_similar += loss_similar.item() * batch_left.size(0)
                running_val_loss_dissimilar += loss_dissimilar.item() * batch_left.size(0)
        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_losses_similar.append(running_val_loss_similar / len(test_loader.dataset))
        val_losses_dissimilar.append(running_val_loss_dissimilar / len(test_loader.dataset))

        # save intermediate ROC
        # if epoch > 0: # and (epoch % 10 == 0 or epoch == num_epochs - 1):
        #     print("Saving ROC curves for epoch", epoch)
        #     make_and_save_roc(model, train_loader, f"rocs/roc_train_epoch{epoch:04}.npz")
        #     make_and_save_roc(model, test_loader, f"rocs/roc_test_epoch{epoch:04}.npz")

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
    # 6. Performance Evaluation: loss components
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.plot(train_losses_similar, label="Training Loss (Duplicates)")
    ax.plot(val_losses_similar, label="Validation Loss (Duplicates)")
    ax.plot(train_losses_dissimilar, label="Training Loss (Non-duplicates)")
    ax.plot(val_losses_dissimilar, label="Validation Loss (Non-duplicates)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend()
    ax.semilogy()
    ax.grid(True)
    pdf.savefig()
    plt.close()

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
    # Plot 1b: ROC curves
    # ----------------------------
    fpr_nn, tpr_nn, _ = roc_curve(all_labels, all_distances)
    fpr_dr, tpr_dr, _ = roc_curve(all_labels, test_dataset_deltaR)
    np.savez("rocs/roc_deltaR.npz", fpr=fpr_dr, tpr=tpr_dr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr_nn, tpr_nn, label='Embedding')
    ax.plot(fpr_dr, tpr_dr, label=r'$\Delta$R')
    ax.set_xlabel('Duplicate efficiency (False Positive Rate)')
    ax.set_ylabel('Non-duplicate efficiency (True Positive Rate)')
    ax.set_title('ROC Curves')
    ax.semilogx()
    ax.set_ylim([0.96, 1.0])
    ax.set_xlim([1e-5, 1])
    # ax.set_ylim([0.99, 1.0])
    # ax.set_xlim([1e-4, 1])
    ax.legend()
    ax.grid(True)
    ax.tick_params(right=True)
    ax.tick_params(top=True, which="minor")
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.13)
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
    with open("train.pkl", "wb") as fi:
        pickle.dump({
            "features": features,
            "embedding_net": embedding_net,
            "test_loader": test_loader,
            "model": model,
        }, fi)

    pdf.close()


def make_deltaR(x_left, x_right):
    eta_l = x_left[:, 1] * 4.0
    phi_l = x_left[:, 2] * 3.1415926
    eta_r = x_right[:, 1] * 4.0
    phi_r = x_right[:, 2] * 3.1415926
    delta_eta = eta_l - eta_r
    delta_phi = dangle(phi_l, phi_r)
    deltaR = np.sqrt(delta_eta**2 + delta_phi**2)
    return deltaR

def dangle(x, y):
    return np.min([(2 * np.pi) - np.abs(x - y), np.abs(x - y)], axis=0)

def make_and_save_roc(model, data_loader, filename):
    model.eval()
    all_distances = []
    all_labels = []
    with torch.no_grad():
        for batch_left, batch_right, batch_label in data_loader:
            distances = model(batch_left, batch_right)
            all_distances.append(distances.cpu().numpy())
            all_labels.append(batch_label.cpu().numpy())
    all_distances = np.concatenate(all_distances).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    fpr, tpr, _ = roc_curve(all_labels, all_distances)
    np.savez(filename, fpr=fpr, tpr=tpr)

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



if __name__ == "__main__":
    main()
