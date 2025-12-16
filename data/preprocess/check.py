import pandas as pd

# Load dataset
dataset_path = "/root/bfr-policy-update/data/raw/uci-2.0.csv"
data = pd.read_csv(dataset_path)

# ------------------------------------------------------------------------------
# 1) Exact duplicate detection (including the label column)
# ------------------------------------------------------------------------------

# Convert each full row (features + label) to a comparable string key
data_str = data.apply(lambda row: "_".join(row.astype(str)), axis=1)

# Map each unique row key to all row indices where it appears
sample_to_indices = {}
for index, sample in data_str.items():
    sample_to_indices.setdefault(sample, []).append(index)

# Keep only keys that occur more than once
duplicates = {k: v for k, v in sample_to_indices.items() if len(v) > 1}

# Print duplicates in a readable format
for duplicate_num, (sample, indices) in enumerate(duplicates.items(), 1):
    print(f"Duplicate_num: {duplicate_num}: {{", end="")
    for i, index in enumerate(indices):
        print(f"{index}: '{sample}'", end="")
        if i < len(indices) - 1:
            print(", ", end="")
    print("}")

# ------------------------------------------------------------------------------
# 2) Conflict detection: identical features with different labels
# ------------------------------------------------------------------------------

# Use all columns except the label as features
features = data.drop(columns=["ACTION"])

# Convert each feature row to a comparable string key
features_str = features.apply(lambda row: "_".join(row.astype(str)), axis=1)

# Track seen feature keys and their observed labels / representative index
feature_to_actions = {}
conflicts = {}

for index, row in data.iterrows():
    feature_key = features_str[index]
    action = row["ACTION"]

    if feature_key in feature_to_actions:
        if action not in feature_to_actions[feature_key]["actions"]:
            # Found a feature-level label conflict
            conflict_num = len(conflicts) + 1
            conflicts.setdefault(conflict_num, {})
            conflicts[conflict_num][feature_to_actions[feature_key]["index"]] = feature_key
            conflicts[conflict_num][index] = feature_key
    else:
        feature_to_actions[feature_key] = {"actions": {action}, "index": index}

# Print conflicts (dictionary form: {row_index: feature_key})
for conflict_num, conflict in conflicts.items():
    print(f"Conflict_num: {conflict_num}: {conflict}")

# ------------------------------------------------------------------------------
# 3) Remove conflicts and save cleaned dataset
# ------------------------------------------------------------------------------

# Rebuild conflict index list (store only row indices)
feature_to_actions = {}
conflicts = {}

for index, row in data.iterrows():
    feature_key = features_str[index]
    action = row["ACTION"]

    if feature_key in feature_to_actions:
        if action not in feature_to_actions[feature_key]["actions"]:
            conflict_num = len(conflicts) + 1
            conflicts.setdefault(conflict_num, [])
            conflicts[conflict_num].append(feature_to_actions[feature_key]["index"])
            conflicts[conflict_num].append(index)
    else:
        feature_to_actions[feature_key] = {"actions": {action}, "index": index}

# Collect all row indices involved in any conflict
conflict_indices = {idx for indices in conflicts.values() for idx in indices}

# Drop conflicts and write the cleaned dataset
data_cleaned = data.drop(index=conflict_indices)
output_path = "/root/bfr-policy-update/data/processed/cleaned_uci-2.0.csv"
data_cleaned.to_csv(output_path, index=False)

print(f"Saved cleaned dataset (conflicting rows removed) to: {output_path}")
