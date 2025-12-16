"""
tune_bfr_hyperparams.py

This module performs a grid search optimization for the BFR (Balance Forgetting and Remembering)
algorithm. It determines the optimal (alpha, beta) hyperparameters—specifically the 
Forgetting Rate (fr) and Retuning Rate (rr)—by minimizing the Minimum Distance to Optimal (MDO)
metric across multiple generated Policy Update Tasks (PUTs).

Workflow:
1. Pre-train a base model on the training set.
2. Generate N random PUT scenarios (Simulating scenarios 1 or 2).
3. Perform grid search over (fr, rr) space.
4. Select parameters that minimize the trade-off between Adaptation (Sample Acc) and Fidelity (AUC).
"""

import pandas as pd
import puts_factory as puts
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from base import DeepNeuralNetwork, LogisticRegressionModel
from unlearners import BFR
import torch
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import scipy.sparse
import time
import copy

def calculate_class_weight(y):
    """
    Compute the positive-class weight for binary labels {0,1}, used in BCEWithLogitsLoss.

    Supported inputs:
      - torch.Tensor
      - pandas.Series
      - numpy.ndarray

    Returns:
      torch.Tensor of shape [1], e.g., tensor([0.73], dtype=torch.float32)
    """
    # (1) Normalize to a NumPy vector
    if torch.is_tensor(y):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    y_np = y_np.astype(int)

    # (2) Count per-class samples (ensure at least two bins)
    class_counts = np.bincount(y_np, minlength=2)
    total_count = len(y_np)

    # (3) Compute weights (equivalent to the original logic)
    weights = total_count / class_counts.astype(float)
    class_weight = weights[1] / (weights[0] + weights[1])

    return torch.tensor([class_weight], dtype=torch.float32)

def balanced_sampling(X_mod, Y_mod, x_sample_mod, y_sample_mod):
    # Compute label ratios in the modified dataset and in the modified sample subset
    y_mod_ratio = np.sum(Y_mod) / len(Y_mod)
    y_sample_mod_ratio = np.sum(y_sample_mod) / len(y_sample_mod)

    # Determine which class to sample and how many samples to draw
    x_sample_mod_length = x_sample_mod.shape[0]

    if y_mod_ratio > y_sample_mod_ratio:
        target_class = 1
        num_samples_to_balance = int(x_sample_mod_length * (y_mod_ratio / (1 - y_mod_ratio)))
    else:
        target_class = 0
        num_samples_to_balance = int(x_sample_mod_length * ((1 - y_mod_ratio) / y_mod_ratio))

    # Sample indices from X_mod / Y_mod by class
    target_indices = np.where(Y_mod == target_class)[0]

    # Do not exceed available samples
    num_samples_to_balance = min(len(target_indices), num_samples_to_balance)

    sampled_target_indices = np.random.choice(target_indices, size=num_samples_to_balance, replace=False)
    sampled_indices = sampled_target_indices

    # Build the auxiliary set
    X_ft = X_mod[sampled_indices]
    Y_ft = Y_mod.iloc[sampled_indices].copy()

    # Merge with modified samples (sparse vertical stack)
    X_ft = scipy.sparse.vstack([X_ft, x_sample_mod])
    Y_ft = pd.concat([Y_ft, y_sample_mod])

    # Sanity check
    if X_ft.shape[0] != Y_ft.shape[0]:
        raise ValueError("The number of samples in X_ft and Y_ft must be the same.")

    return X_ft, Y_ft

def sample_put_from_train(X_train, Y_train, one_hot_encoder,
                          modification_type, sample_num):
    """
    Construct one PUT scenario from (X_train, Y_train):
      - Randomly select `sample_num` samples and apply the chosen modification operator
      - Return the modified dataset and the affected sample pairs
      - Outputs are one-hot encoded using the provided encoder

    Returns:
      X_mod_enc, Y_mod,
      x_sample_mod_enc, y_sample_mod,
      x_sample_enc, y_sample,
      indices_sample
    """
    # Apply the selected PUT operator (kept unchanged)
    if modification_type == 'mod_features_and_labels':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = \
            puts.modify_features_and_labels(X_train, Y_train, n=sample_num, k=1, p=3)
    elif modification_type == 'mod_features_only':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = \
            puts.modify_features(X_train, Y_train, n=sample_num, k=1, p=3)
    elif modification_type == 'mod_labels_only':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = \
            puts.modify_labels(X_train, Y_train, n=sample_num)
    else:
        raise ValueError("Invalid modification type")

    # One-hot encode using the shared encoder
    X_mod_enc = one_hot_encoder.transform(X_mod)
    x_sample_enc = one_hot_encoder.transform(x_sample)
    x_sample_mod_enc = one_hot_encoder.transform(x_sample_mod)

    return (
        X_mod_enc,
        Y_mod,
        x_sample_mod_enc,
        y_sample_mod,
        x_sample_enc,
        y_sample,
        indices_sample,
    )

def compute_bfr_metrics(models, X_test, Y_test, X_sample_mod, Y_sample_mod):
    """
    Extract two metrics from each model:
      - AUC on (X_test, Y_test)
      - Sample Accuracy on (X_sample_mod, Y_sample_mod)
    """
    performance_dict = {}
    for model_name, model in models.items():
        performance = model.evaluate_auc_sample(
            X_test, Y_test, X_sample_mod, Y_sample_mod
        )
        performance_dict[model_name] = performance
    return performance_dict

def tune_bfr_hyperparams(num_runs, num_points, sample_num,
                         dataset_path, modification_type,
                         data_type, base_select='LR'):
    """
    Grid search over (forgetting_rate, retuning_rate) for BFR.

    Args:
      - num_runs: number of PUT scenarios (J)
      - num_points: grid resolution per dimension
      - sample_num: number of modified policies per PUT (K)

    Returns:
      best_fr, best_rr, results_df
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BFR tuning] device = {device}")

    # ---------------- 1) Load data and split train/test ----------------
    data = pd.read_csv(dataset_path)
    Y = data['ACTION']
    X = data.drop('ACTION', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=40
    )
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    # One-hot encoder shared across train/test
    one_hot_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    one_hot_encoder = one_hot_enc.fit(X)

    X_train_enc = one_hot_encoder.transform(X_train)
    X_test_enc = one_hot_encoder.transform(X_test)

    # Convert to tensors for training/evaluation
    X_train_enc_t = torch.tensor(
        X_train_enc.toarray(), dtype=torch.float32
    ).to(device)
    Y_train_t = torch.tensor(
        Y_train.to_numpy(), dtype=torch.float32
    ).to(device)
    X_test_enc_t = torch.tensor(
        X_test_enc.toarray(), dtype=torch.float32
    ).to(device)
    Y_test_t = torch.tensor(
        Y_test.to_numpy(), dtype=torch.float32
    ).to(device)

    # ---------------- 2) Define the hyperparameter search range ----------------
    if base_select == 'LR':
        if data_type == 'kaggle':
            fr_max = 4.16
            rr_max = 1.73
        elif data_type == 'uci':
            fr_max = 4.53
            rr_max = 2.39
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")
    elif base_select == 'DNN':
        if data_type == 'kaggle':
            fr_max = 0.04
            rr_max = 0.014
        elif data_type == 'uci':
            fr_max = 0.08
            rr_max = 0.024
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")
    else:
        raise ValueError(f"Unsupported base model type: {base_select}")

    forgetting_rate_values = np.linspace(0, fr_max, num_points)
    retuning_rate_values = np.linspace(0, rr_max, num_points)

    # ---------------- 3) Train the base model theta* once ----------------
    input_dim = X_train_enc_t.shape[1]
    if base_select == 'LR':
        base_model = LogisticRegressionModel(input_dim).to(device)
    else:
        base_model = DeepNeuralNetwork(input_dim).to(device)

    base_model.train_model(X_train_enc_t, Y_train_t)
    base_state = copy.deepcopy(base_model.state_dict())
    print("[BFR tuning] base model trained.")

    start_tuning = time.time()

    # ---------------- 4) Pre-generate num_runs PUT scenarios ----------------
    scenarios = []
    for j in range(num_runs):
        (
            X_mod_enc,
            Y_mod,
            x_sample_mod_enc,
            y_sample_mod,
            x_sample_enc,
            y_sample,
            indices_sample,
        ) = sample_put_from_train(
            X_train, Y_train, one_hot_encoder,
            modification_type, sample_num
        )

        sampling_start = time.time()
        X_rt2, Y_rt2 = balanced_sampling(
            X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod
        )
        sampling_end = time.time()
        sampling_time = sampling_end - sampling_start

        #X_mod_enc_t = torch.tensor(
        #    X_mod_enc.toarray(), dtype=torch.float32
        #).to(device)
        Y_mod_t = torch.tensor(
            Y_mod.to_numpy(), dtype=torch.float32
        ).to(device)
        x_sample_mod_enc_t = torch.tensor(
            x_sample_mod_enc.toarray(), dtype=torch.float32
        ).to(device)
        y_sample_mod_t = torch.tensor(
            y_sample_mod.to_numpy(), dtype=torch.float32
        ).to(device)
        x_sample_enc_t = torch.tensor(
            x_sample_enc.toarray(), dtype=torch.float32
        ).to(device)
        y_sample_t = torch.tensor(
            y_sample.to_numpy(), dtype=torch.float32
        ).to(device)
        X_rt2_t = torch.tensor(
            X_rt2.toarray(), dtype=torch.float32
        ).to(device)
        Y_rt2_t = torch.tensor(
            Y_rt2.values, dtype=torch.float32
        ).to(device)

        weights = calculate_class_weight(Y_mod_t)

        scenarios.append(
            dict(
                #X_mod=X_mod_enc_t,
                Y_mod=Y_mod_t,
                x_sample_mod=x_sample_mod_enc_t,
                y_sample_mod=y_sample_mod_t,
                x_sample=x_sample_enc_t,
                y_sample=y_sample_t,
                X_rt2=X_rt2_t,
                Y_rt2=Y_rt2_t,
                weights=weights,
                sampling_time=sampling_time,
            )
        )

    # ---------------- 5) Evaluate each (fr, rr) over scenarios and average MDO ----------------
    best_mean_mdo = float("inf")
    best_fr = None
    best_rr = None

    records = []

    for fr in forgetting_rate_values:
        for rr in retuning_rate_values:
            mdo_list = []

            for j, scen in enumerate(tqdm(scenarios, desc="Scenarios", leave=False)):
                # Clone a fresh model from theta* for each scenario
                if base_select == 'LR':
                    model_j = LogisticRegressionModel(input_dim).to(device)
                else:
                    model_j = DeepNeuralNetwork(input_dim).to(device)
                model_j.load_state_dict(base_state)

                bfr = BFR(
                    model_j,
                    base_select,
                    fr,              # forgetting_rate
                    rr,              # retuning_rate
                    scen["weights"], # class weights
                )

                start_t = time.time()
                bfr_model = bfr.training(
                    #scen["X_mod"],
                    #scen["Y_mod"],
                    scen["x_sample"],
                    scen["y_sample"],
                    scen["x_sample_mod"],
                    scen["y_sample_mod"],
                    scen["X_rt2"],
                    scen["Y_rt2"],
                )
                end_t = time.time()
                run_time = end_t - start_t + scen["sampling_time"]

                perf = compute_bfr_metrics(
                    {"BFR": bfr_model},
                    X_test_enc_t,
                    Y_test_t,
                    scen["x_sample_mod"],
                    scen["y_sample_mod"],
                )
                auc = perf["BFR"]["AUC"]
                acc = perf["BFR"]["Sample Accuracy"]

                mdo = float(np.sqrt((1.0 - acc) ** 2 + (1.0 - auc) ** 2))
                mdo_list.append(mdo)

                records.append(
                    dict(
                        Model="BFR",
                        ForgettingRate=fr,
                        RetuningRate=rr,
                        Scenario=j + 1,
                        Test_AUC=auc,
                        Sample_Accuracy=acc,
                        MDO=mdo,
                        Run_Time=run_time,
                    )
                )

            mean_mdo = float(np.mean(mdo_list))

            if mean_mdo < best_mean_mdo:
                best_mean_mdo = mean_mdo
                best_fr = fr
                best_rr = rr

    results_df = pd.DataFrame(records)
    end_tuning = time.time()
    elapsed_time = end_tuning - start_tuning

    print(
        f"Best hyperparameters: forgetting_rate = {best_fr:.4f}, "
        f"remembering_rate = {best_rr:.4f}, "
        f"mean MDO = {best_mean_mdo:.4f}, "
        f"tuning runtime = {elapsed_time:.4f} s"
    )

    return best_fr, best_rr, results_df


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Configuration ---
    # Dataset paths
    UCI_DATASET_PATH = "/root/5_MLForAC/dataSet/cleaned_uci-2.0.csv"
    KAGGLE_DATASET_PATH = "/root/5_MLForAC/dataSet/train.csv"

    # Dataset selection: (path, name)
    CURRENT_DATASET = (UCI_DATASET_PATH, "uci")

    # Experiment settings
    MODIFICATION_TYPE = "mod_labels_only"  # Options: 'mod_features_and_labels', 'mod_labels_only'
    NUM_RUNS = 10
    NUM_POINTS = 10
    SAMPLE_NUM = 80
    BASE_MODEL = "DNN"                     # Options: 'DNN', 'LR'

    # --- Execution ---
    dataset_path, dataset_name = CURRENT_DATASET

    tune_bfr_hyperparams(
        num_runs=NUM_RUNS,
        num_points=NUM_POINTS,
        sample_num=SAMPLE_NUM,
        dataset_path=dataset_path,
        modification_type=MODIFICATION_TYPE,
        data_type=dataset_name,
        base_select=BASE_MODEL
    )
