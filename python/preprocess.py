#!/usr/bin/env python

import uproot
import numpy as np
import random


BRANCHES_LIST = [
    'tc_isDuplicate',
    'tc_pt',
    'tc_eta',
    'tc_phi',
    'tc_type',
    'tc_matched_simIdx',
]


def preprocess_data(file_path: str):
    branches = load_root_file(file_path, BRANCHES_LIST, True)

    print("Printing basic branch info:")
    print(len(branches['tc_matched_simIdx']))
    print(branches['tc_matched_simIdx'][0])
    print(branches['tc_matched_simIdx'][1])
    print(branches['tc_matched_simIdx'][2])
    print(len(branches['tc_matched_simIdx'][0]))
    print(len(branches['tc_matched_simIdx'][1]))
    print(len(branches['tc_matched_simIdx'][2]))
    print(branches["tc_pt"][0])
    print(len(branches["tc_pt"][0]))
    print(len(branches["tc_pt"][1]))
    print(len(branches["tc_pt"][2]))

    print("")
    print("Checking how many sim hits are matched to tracks:")
    num_matched = {}
    for ev in range(len(branches['tc_matched_simIdx'])):
        # print(f"On event {ev}")
        for tk in range(len(branches['tc_matched_simIdx'][ev])):
            n_matched = len(branches['tc_matched_simIdx'][ev][tk])
            num_matched[n_matched] = num_matched.get(n_matched, 0) + 1
    print(num_matched)


    features = []
    n_events = len(branches['tc_pt'])
    for i in range(n_events):
        # Create an array of shape (n_tracks, 4) for each event.
        event_features = np.column_stack((
            np.log10(branches['tc_pt'][i]),
            (branches['tc_eta'][i]/4.0),
            (branches['tc_phi'][i]/3.1415926),
            (branches['tc_type'][i]-6.0)/2.0
        ))
        features.append(event_features)

    # Extract plain sim indices from the STLVector branch.
    sim_indices = extract_sim_indices(branches['tc_matched_simIdx'])

    # Generate balanced pairs from each event.
    x_left, x_right, y = create_pairs_balanced(features, sim_indices,
                                                max_duplicate_pairs_per_event=30,
                                                max_nonduplicate_pairs_per_event=270)

    print("Total pairs generated:", len(y))
    print("Duplicate pairs (label 0):", np.sum(y == 0))
    print("Non-duplicate pairs (label 1):", np.sum(y == 1))

    return branches, features, sim_indices, x_left, x_right, y


def load_root_file(file_path, branches=None, print_branches=False):
    all_branches = {}
    with uproot.open(file_path) as file:
        tree = file["tree"]
        # Load all ROOT branches into array if not specified
        if branches is None:
            branches = tree.keys()
        # Option to print the branch names
        if print_branches:
            print("Branches:", tree.keys())
        # Each branch is added to the dictionary
        for branch in branches:
            try:
                all_branches[branch] = (tree[branch].array(library="np"))
            except uproot.KeyInFileError as e:
                print(f"KeyInFileError: {e}")
        # Number of events in file
        all_branches['event'] = tree.num_entries
    return all_branches


def extract_sim_indices(sim_idx_branch):
    """
    Extracts sim index integers from an array of STLVector objects.
    Each event in sim_idx_branch is an STLVector of tracks,
    and each track is itself an STLVector containing a single integer.
    """
    extracted = []
    for event in sim_idx_branch:
        event_indices = []
        for track in event:
            if len(track) > 0:
                event_indices.append(track[0])
            else:
                event_indices.append(None)
        extracted.append(event_indices)
    return extracted


def create_pairs_balanced(features, sim_indices, max_duplicate_pairs_per_event=20, max_nonduplicate_pairs_per_event=90):
    """
    For each event, generate duplicate pairs by grouping tracks with the same sim index.
    Also, generate a set of non-duplicate pairs (tracks with different sim indices).
    This routine returns a more balanced dataset.
    """
    duplicate_left = []
    duplicate_right = []
    nonduplicate_left = []
    nonduplicate_right = []

    for event_features, event_sim in zip(features, sim_indices):
        n_tracks = event_features.shape[0]
        # Build a dictionary mapping sim index to track indices.
        sim_dict = {}
        for idx, sim in enumerate(event_sim):
            if sim is not None:
                sim_dict.setdefault(sim, []).append(idx)

        # Duplicate pairs: for each sim index that appears more than once.
        dup_pairs = []
        for sim, indices in sim_dict.items():
            if len(indices) > 1:
                # Form all pairs among indices
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        dup_pairs.append((indices[i], indices[j]))
        # Sample a maximum number per event if needed.
        if len(dup_pairs) > max_duplicate_pairs_per_event:
            dup_pairs = random.sample(dup_pairs, max_duplicate_pairs_per_event)
        for i, j in dup_pairs:
            duplicate_left.append(event_features[i])
            duplicate_right.append(event_features[j])

        # Non-duplicate pairs: pairs with different sim indices.
        nondup_pairs = []
        for i in range(n_tracks):
            for j in range(i+1, n_tracks):
                # If either sim index is None or they differ, count as non-duplicate.
                if event_sim[i] != event_sim[j]:
                    nondup_pairs.append((i, j))
        if len(nondup_pairs) > max_nonduplicate_pairs_per_event:
            nondup_pairs = random.sample(nondup_pairs, max_nonduplicate_pairs_per_event)
        for i, j in nondup_pairs:
            nonduplicate_left.append(event_features[i])
            nonduplicate_right.append(event_features[j])

    # Combine duplicate and non-duplicate pairs.
    left = np.concatenate([np.array(duplicate_left), np.array(nonduplicate_left)], axis=0)
    right = np.concatenate([np.array(duplicate_right), np.array(nonduplicate_right)], axis=0)
    labels = np.concatenate([np.zeros(len(duplicate_left)), np.ones(len(nonduplicate_left))], axis=0)
    return left, right, labels


if __name__ == "__main__":
    # Example usage
    branches, features, sim_indices, x_left, x_right, y = preprocess_data("../data/LSTNtuple.root")
