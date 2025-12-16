"""
puts_factory.py

This module generates Policy Update Tasks (PUTs) for the BFR framework. 
It constructs synthetic update scenarios by perturbing features and/or flipping 
authorization labels from the original dataset.

It implements the specific logic for:
- [cite_start]Scenario 1: Authorization Revocation/Granting (Label flipping only) [cite: 123]
- [cite_start]Scenario 2: Policy Modification (Feature perturbation + Label flipping) [cite: 124]

Note: Public function signatures are preserved for backward compatibility.
"""

import pandas as pd
import numpy as np
import random

def modify_features(X, Y, n, k, p):
    # Ensure n, k, and p are within valid ranges
    n = min(n, len(X))
    k = min(max(1, k), X.shape[1])  # At least 1 feature, at most all features
    p = min(max(k, p), X.shape[1])  # At least k features, at most all features

    # Create copies of the original datasets
    X_mod = X.copy()
    Y_mod = Y.copy()

    # Randomly select n samples
    indices = np.random.choice(X.index, size=n, replace=False)
    x_sample_mod = X.loc[indices].copy()
    y_sample_mod = Y.loc[indices].copy()

    # Ensure y_sample_mod contains at least two classes
    unique_classes = np.unique(y_sample_mod)
    if len(unique_classes) == 1:
        # If only one class appears, remove one sample
        remove_idx = np.random.choice(indices, size=1, replace=False)
        x_sample_mod = x_sample_mod.drop(remove_idx)
        y_sample_mod = y_sample_mod.drop(remove_idx)

        # Remove the dropped index from indices
        indices = np.setdiff1d(indices, remove_idx)

        # Add one sample from the opposite class
        different_class_indices = Y[Y != unique_classes[0]].index
        new_idx = np.random.choice(different_class_indices, size=1, replace=False)
        x_sample_mod = pd.concat([x_sample_mod, X.loc[new_idx]])
        y_sample_mod = pd.concat([y_sample_mod, Y.loc[new_idx]])

        # Append the new index
        indices = np.append(indices, new_idx)

    # Save original values of the selected samples
    x_sample = X.loc[indices].copy()
    y_sample = Y.loc[indices].copy()

    # Modify features for the selected samples
    for index in indices:
        # Randomly choose how many features to modify for this sample
        num_features_to_modify = random.randint(k, p)

        # Randomly select which features to modify
        features_to_modify = random.sample(list(X.columns), num_features_to_modify)

        # Apply feature modifications
        for feature in features_to_modify:
            # Draw a new value from the feature's observed values
            new_value = np.random.choice(X[feature].unique())
            X_mod.at[index, feature] = new_value
            x_sample_mod.at[index, feature] = new_value

        # Note: labels are unchanged (Y_mod and y_sample_mod)

    return X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices

def modify_labels(X, Y, n):
    # Ensure n is within a valid range
    n = min(n, len(X))

    # Create copies of the original datasets
    X_mod = X.copy()
    Y_mod = Y.copy()

    # Randomly select n samples
    indices = np.random.choice(X.index, size=n, replace=False)
    x_sample_mod = X.loc[indices].copy()
    y_sample_mod = Y.loc[indices].copy()

    # Ensure y_sample_mod contains at least two classes
    unique_classes = np.unique(y_sample_mod)
    if len(unique_classes) == 1:
        # If only one class appears, remove one sample
        remove_idx = np.random.choice(indices, size=1, replace=False)
        x_sample_mod = x_sample_mod.drop(remove_idx)
        y_sample_mod = y_sample_mod.drop(remove_idx)

        # Remove the dropped index from indices
        indices = np.setdiff1d(indices, remove_idx)

        # Add one sample from the opposite class
        different_class_indices = Y[Y != unique_classes[0]].index
        new_idx = np.random.choice(different_class_indices, size=1, replace=False)
        x_sample_mod = pd.concat([x_sample_mod, X.loc[new_idx]])
        y_sample_mod = pd.concat([y_sample_mod, Y.loc[new_idx]])

        # Append the new index
        indices = np.append(indices, new_idx)

    # Save original values of the selected samples
    x_sample = X.loc[indices].copy()
    y_sample = Y.loc[indices].copy()

    # Flip labels for the selected samples
    for index in indices:
        Y_mod.at[index] = 1 if Y.at[index] == 0 else 0
        y_sample_mod.at[index] = 1 if y_sample_mod.at[index] == 0 else 0

    #utils.print_label_proportions(y_sample_mod)
    return X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices

def modify_features_and_labels(X, Y, n, k, p):
    # Ensure n, k, and p are within valid ranges
    n = min(n, len(X))
    k = min(max(1, k), X.shape[1])  # At least 1 feature, at most all features
    p = min(max(k, p), X.shape[1])  # At least k features, at most all features

    # Create copies of the original datasets
    X_mod = X.copy()
    Y_mod = Y.copy()

    # Randomly select n samples
    indices = np.random.choice(X.index, size=n, replace=False)
    x_sample_mod = X.loc[indices].copy()
    y_sample_mod = Y.loc[indices].copy()

    # Ensure y_sample_mod contains at least two classes
    unique_classes = np.unique(y_sample_mod)
    if len(unique_classes) == 1:
        # If only one class appears, remove one sample
        remove_idx = np.random.choice(indices, size=1, replace=False)
        x_sample_mod = x_sample_mod.drop(remove_idx)
        y_sample_mod = y_sample_mod.drop(remove_idx)

        # Remove the dropped index from indices
        indices = np.setdiff1d(indices, remove_idx)

        # Add one sample from the opposite class
        different_class_indices = Y[Y != unique_classes[0]].index
        new_idx = np.random.choice(different_class_indices, size=1, replace=False)
        x_sample_mod = pd.concat([x_sample_mod, X.loc[new_idx]])
        y_sample_mod = pd.concat([y_sample_mod, Y.loc[new_idx]])

        # Append the new index
        indices = np.append(indices, new_idx)
        
    # Save original values of the selected samples
    x_sample = X.loc[indices].copy()
    y_sample = Y.loc[indices].copy()
    
    # Modify features and labels for the selected samples
    for index in indices:
        # Randomly choose how many features to modify for this sample
        num_features_to_modify = random.randint(k, p)

        # Randomly select which features to modify
        features_to_modify = random.sample(list(X.columns), num_features_to_modify)

        # Apply feature modifications
        for feature in features_to_modify:
            # Draw a new value from the feature's observed values
            new_value = np.random.choice(X[feature].unique())
            X_mod.at[index, feature] = new_value
            x_sample_mod.at[index, feature] = new_value

        # Flip label
        Y_mod.at[index] = 1 if Y.at[index] == 0 else 0
        y_sample_mod.at[index] = 1 if y_sample_mod.at[index] == 0 else 0

    #print(y_sample_mod)
    return X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices


def modify_features_and_labels_balanced(X, Y, n, k, p):
    # Ensure n is even to enforce class balance
    if n % 2 != 0:
        raise ValueError("n must be an even number to ensure class balance.")
    
    # Ensure n, k, and p are within valid ranges
    n = min(n, len(X))
    k = min(max(1, k), X.shape[1])  # At least 1 feature, at most all features
    p = min(max(k, p), X.shape[1])  # At least k features, at most all features
    
    # Create copies of the original datasets
    X_mod = X.copy()
    Y_mod = Y.copy()
    
    # Randomly select n/2 samples from each class
    indices_0 = np.random.choice(Y[Y == 0].index, size=n//2, replace=False)
    indices_1 = np.random.choice(Y[Y == 1].index, size=n//2, replace=False)
    indices = np.concatenate([indices_0, indices_1])
    
    # Save original values of the selected samples
    x_sample = X.loc[indices].copy()
    y_sample = Y.loc[indices].copy()
    x_sample_mod = X.loc[indices].copy()
    y_sample_mod = Y.loc[indices].copy()
    
    # Modify features and labels for the selected samples
    for index in indices:
        # Randomly choose how many features to modify for this sample
        num_features_to_modify = random.randint(k, p)
        
        # Randomly select which features to modify
        features_to_modify = random.sample(list(X.columns), num_features_to_modify)
        
        # Apply feature modifications
        for feature in features_to_modify:
            # Draw a new value from the feature's observed values
            new_value = np.random.choice(X[feature].unique())
            X_mod.at[index, feature] = new_value
            x_sample_mod.at[index, feature] = new_value
            
        # Flip label
        new_label = 1 if Y.at[index] == 0 else 0
        Y_mod.at[index] = new_label
        y_sample_mod.at[index] = new_label
    
    return X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices

def sample_balanced_data(X, Y, n):
    # Ensure n is even to balance the two classes
    n = n if n % 2 == 0 else n - 1

    # Get indices per class
    indices_0 = Y[Y == 0].index
    indices_1 = Y[Y == 1].index

    # Randomly select n/2 samples from each class
    sample_indices_0 = np.random.choice(indices_0, size=n//2, replace=False)
    sample_indices_1 = np.random.choice(indices_1, size=n//2, replace=False)

    # Combine indices
    combined_indices = np.concatenate([sample_indices_0, sample_indices_1])

    # Extract samples
    x_sample_mod = X.loc[combined_indices].copy()
    y_sample_mod = Y.loc[combined_indices].copy()

    return x_sample_mod, y_sample_mod, combined_indices

def modify_features_and_labels_balance(X, Y, n, k, p):
    # Ensure n, k, and p are within valid ranges
    n = min(n, len(X))
    k = min(max(1, k), X.shape[1])  # At least 1 feature, at most all features
    p = min(max(k, p), X.shape[1])  # At least k features, at most all features

    # Create copies of the original datasets
    X_mod = X.copy()
    Y_mod = Y.copy()

    x_sample_mod, y_sample_mod, indices = sample_balanced_data(X, Y, n)
        
    # Save original values of the selected samples
    x_sample = X.loc[indices].copy()
    y_sample = Y.loc[indices].copy()
    
    # Modify features and labels for the selected samples
    for index in indices:
        # Randomly choose how many features to modify for this sample
        num_features_to_modify = random.randint(k, p)

        # Randomly select which features to modify
        features_to_modify = random.sample(list(X.columns), num_features_to_modify)

        # Apply feature modifications
        for feature in features_to_modify:
            # Draw a new value from the feature's observed values
            new_value = np.random.choice(X[feature].unique())
            X_mod.at[index, feature] = new_value
            x_sample_mod.at[index, feature] = new_value

        # Flip label
        Y_mod.at[index] = 1 if Y.at[index] == 0 else 0
        y_sample_mod.at[index] = 1 if y_sample_mod.at[index] == 0 else 0

    return X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices


def modify_labels_balance(X, Y, n):
    # Ensure n is within a valid range
    n = min(n, len(X))

    # Create copies of the original datasets
    X_mod = X.copy()
    Y_mod = Y.copy()

    x_sample_mod, y_sample_mod, indices = sample_balanced_data(X, Y, n)

    # Save original values of the selected samples
    x_sample = X.loc[indices].copy()
    y_sample = Y.loc[indices].copy()
    
    # Flip labels for the selected samples
    for index in indices:
        # Sanity check before modification
        if (Y_mod.at[index] != y_sample_mod.at[index]):
            print("False")
        Y_mod.at[index] = 1 if Y.at[index] == 0 else 0
        y_sample_mod.at[index] = 1 if y_sample.at[index] == 0 else 0

    #utils.print_label_proportions(y_sample_mod)
    return X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices
