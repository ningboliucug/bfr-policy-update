"""
Policy Update Unlearning Benchmark Framework (BFR-MLBAC)

This script executes a comprehensive evaluation of various machine unlearning 
methods (BFR, SISA, First-Order, Fine-Tuning) in the context of access control 
policy updates. It handles data preprocessing, model training, and performance 
benchmarking across fidelity, adaptability, and efficiency metrics.

Reference: 
    "Balance Forgetting and Remembering: An Extension of Machine Unlearning for 
     Policy Updates in Machine Learning-Based Access Control"
"""

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse import vstack, csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from copy import deepcopy
import itertools
import warnings
import csv
import time
import sys

import puts_factory as puts
from base import DeepNeuralNetwork, LogisticRegressionModel
from unlearners import Retrainer, First_Order, Fine_Tuning, BFR, SISA_DNN, SISA_LR


def stratified_sample(X_test, Y_test, X_sample_mod, Y_sample_mod, factor=50):
    # Compute label counts in the modified-sample set
    unique_labels, label_counts = np.unique(Y_sample_mod.cpu().numpy(), return_counts=True)

    # Total number of samples to draw
    total_samples = len(Y_sample_mod) * factor

    # Target sample counts per label after stratification
    label_ratios = label_counts / len(Y_sample_mod)
    target_label_counts = (label_ratios * total_samples).astype(int)

    # Ensure the total matches exactly
    if target_label_counts.sum() < total_samples:
        target_label_counts[0] += total_samples - target_label_counts.sum()

    # Perform stratified resampling from the test set
    X_sampled_list = []
    Y_sampled_list = []

    for label, count in zip(unique_labels, target_label_counts):
        X_label = X_test[Y_test == label]
        Y_label = Y_test[Y_test == label]
        X_sampled, Y_sampled = resample(
            X_label.cpu().numpy(),
            Y_label.cpu().numpy(),
            n_samples=count,
            random_state=42
        )
        X_sampled_list.append(torch.tensor(X_sampled, dtype=torch.float32).to(X_test.device))
        Y_sampled_list.append(torch.tensor(Y_sampled, dtype=torch.float32).to(Y_test.device))

    X_sampled_final = torch.cat(X_sampled_list, dim=0)
    Y_sampled_final = torch.cat(Y_sampled_list, dim=0)

    return X_sampled_final, Y_sampled_final


def calculate_class_weight(Y_train):
    # Compute class weights for binary labels {0,1}
    class_counts = np.bincount(Y_train.cpu().numpy().astype(int))
    total_count = len(Y_train)
    weights = total_count / class_counts
    class_weight = weights[1] / (weights[0] + weights[1])
    return torch.tensor([class_weight], dtype=torch.float32)


def balanced_sampling(X_mod, Y_mod, x_sample_mod, y_sample_mod):
    # Estimate label ratios in (1) modified training set and (2) modified sample set
    y_mod_ratio = np.sum(Y_mod) / len(Y_mod)
    y_sample_mod_ratio = np.sum(y_sample_mod) / len(y_sample_mod)

    # Decide which class to sample to better match the target ratio
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

    # Build the balanced set
    X_ft = X_mod[sampled_indices]
    Y_ft = Y_mod.iloc[sampled_indices].copy()

    # Merge with modified samples and keep sparse representation for now
    X_ft = scipy.sparse.vstack([X_ft, x_sample_mod])
    Y_ft = pd.concat([Y_ft, y_sample_mod])

    # Sanity check
    if X_ft.shape[0] != Y_ft.shape[0]:
        raise ValueError("The number of samples in X_ft and Y_ft must be the same.")

    return X_ft, Y_ft


def data_engineering(dataset_path, modification_type, sample_num):
    # Load dataset
    data = pd.read_csv(dataset_path)
    Y = data['ACTION']
    X = data.drop('ACTION', axis=1)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    # Apply policy-update simulation (sample modification)
    if modification_type == 'mod_features_and_labels':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = puts.modify_features_and_labels(
            X_train, Y_train, n=sample_num, k=1, p=3
        )
        # X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = puts.modify_features_and_labels_balanced(
        #     X_train, Y_train, n=sample_num, k=1, p=3
        # )
    elif modification_type == 'mod_features_only':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = puts.modify_features(
            X_train, Y_train, n=sample_num, k=1, p=3
        )
    elif modification_type == 'mod_labels_only':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = puts.modify_labels(
            X_train, Y_train, n=sample_num
        )
    else:
        raise ValueError("Invalid modification type")

    # One-hot encoding
    one_hot_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    one_hot_encoder = one_hot_enc.fit(X)
    X_test_enc = one_hot_encoder.transform(X_test)
    X_train_enc = one_hot_encoder.transform(X_train)
    X_mod_enc = one_hot_encoder.transform(X_mod)
    x_sample_enc = one_hot_encoder.transform(x_sample)
    x_sample_mod_enc = one_hot_encoder.transform(x_sample_mod)

    # Build a unified probability-evaluation set:
    # sample a subset from X_test_enc and append x_sample_mod_enc
    num_samples = int(x_sample_mod_enc.shape[0] * 1)
    random_indices = np.random.choice(X_test_enc.shape[0], num_samples, replace=False)
    X_test_sampled = X_test_enc[random_indices, :]
    X_proba = vstack([X_test_sampled, x_sample_mod_enc])
    y_proba = np.concatenate([Y_test[random_indices], y_sample_mod])

    return (
        X_train_enc, Y_train, X_test_enc, Y_test,
        X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod,
        x_sample_enc, y_sample, indices_sample,
        X_proba, y_proba
    )


def get_all_performance(models, retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba):
    performance_dict = {}

    for model_name, model in models.items():
        # Evaluate each model via its evaluate_model() method
        performance = model.evaluate_model(retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba)
        performance_dict[model_name] = performance

    return performance_dict


def run_comprehensive_evaluation(
    num_runs, num_points, sample_num, n_shards,
    dataset_path, modification_type, data_type,
    base_select, model='all'
):
    result_file = "performance_" + data_type + "_" + modification_type + "_" + base_select + "_" + str(sample_num) + ".csv"
    print(result_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 1. Define Hyperparameter Search Space
    # Note: The upper bounds for hyperparameters are empirically determined via 'tune_bft_hyperparams.py'.
    # These bounds are calibrated to cover the full "Effective Trade-off Region"—ranging from 
    # High Fidelity/Low Adaptation (near 0) to High Adaptation/Lower Fidelity (near max).
    # Searching beyond these bounds yields diminishing returns or model performance collapse.
    if base_select == 'LR':
        if data_type == 'kaggle':
            fo_tao_max = 40
            ft_C_max = 0.1404 * 1.3          # S1: 0.1404
            fr_max = 2.288 * 1.4 * 1.3       # S1: 2.288
            rr_max = 0.132 * 3 * 1.5 * 1.2   # S1: 0.132
        elif data_type == 'uci':
            fo_tao_max = 40 * 1.01
            ft_C_max = 1 * 1.1
            fr_max = 3 * 1.2 * 1.2 * 1.05
            rr_max = 1 * 1.5 * 1.1 * 1.1 * 1.2 * 1.1
    elif base_select == 'DNN':
        if data_type == 'kaggle':
            fo_tao_max = 3.5
            ft_C_max = 0.0144
            fr_max = 0.04
            rr_max = 0.014
        elif data_type == 'uci':
            fo_tao_max = 2.6
            ft_C_max = 0.04
            fr_max = 0.05 * 1.6             # S1: 0.05; S2: 0.05 * 1.2
            rr_max = 0.015 * 1.6            # S1: 0.015; S2: 0.015 * 1.5

    linear_values = np.linspace(0, 1, num_points)
    fo_tao_values = np.linspace(0, fo_tao_max, num_points)
    ft_C_values = np.linspace(0, ft_C_max, num_points)
    forgetting_rate_values = np.linspace(0, fr_max, num_points)
    retuning_rate_values = np.linspace(0, rr_max, num_points)

    bfr_param_pairs = list(itertools.product(forgetting_rate_values, retuning_rate_values))

    ppc_models = ['BFR', 'First_Order', 'Fine_Tuning']

    # Accumulate all runs into one table
    cumulative_results = pd.DataFrame(columns=[
        "Model", "Test_AUC", "FPR", "Test_Accuracy", "Mod_Sample_Accuracy",
        "MIA AUC.", "JS Div.", "Parameter", "Run", "Run_Time"
    ])

    # Outer loop: repeat experiments for robustness
    for run in tqdm(range(num_runs), desc="Total Running"):
        print(f"Run {run+1}/{num_runs}")
        run_results = pd.DataFrame(columns=[
            "Model", "Test_AUC", "FPR", "Test_Accuracy", "Mod_Sample_Accuracy",
            "MIA AUC.", "JS Div.", "Parameter", "Run", "Run_Time"
        ])

        # Data preparation
        X_train_enc, Y_train, X_test_enc, Y_test, X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod, x_sample_enc, y_sample, indices_sample, X_proba, y_proba = data_engineering(
            dataset_path, modification_type, sample_num
        )

        # For SISA: track changed samples
        changed_data = {idx: (x_sample_mod_enc[i].toarray(), y_sample_mod.iloc[i]) for i, idx in enumerate(indices_sample)}

        # Build an auxiliary balanced set for BFR retuning
        sampling_start_time = time.time()
        X_rt2, Y_rt2 = balanced_sampling(X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod)
        sampling_end_time = time.time()
        sampling_time = sampling_end_time - sampling_start_time

        # Move data to torch tensors
        X_train_enc = torch.tensor(X_train_enc.toarray(), dtype=torch.float32).to(device)
        Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.float32).to(device)
        X_test_enc = torch.tensor(X_test_enc.toarray(), dtype=torch.float32).to(device)
        Y_test = torch.tensor(Y_test.to_numpy(), dtype=torch.float32).to(device)
        X_mod_enc = torch.tensor(X_mod_enc.toarray(), dtype=torch.float32).to(device)
        Y_mod = torch.tensor(Y_mod.to_numpy(), dtype=torch.float32).to(device)
        x_sample_mod_enc = torch.tensor(x_sample_mod_enc.toarray(), dtype=torch.float32).to(device)
        y_sample_mod = torch.tensor(y_sample_mod.to_numpy(), dtype=torch.float32).to(device)
        x_sample_enc = torch.tensor(x_sample_enc.toarray(), dtype=torch.float32).to(device)
        y_sample = torch.tensor(y_sample.to_numpy(), dtype=torch.float32).to(device)
        X_proba = torch.tensor(X_proba.toarray(), dtype=torch.float32).to(device)
        X_rt2_t = torch.tensor(X_rt2.toarray(), dtype=torch.float32)
        Y_rt2_t = torch.tensor(Y_rt2.values, dtype=torch.float32)

        # Train base model
        static_models = {}
        dynamic_models = {}
        runtime = {}
        input_dim = X_train_enc.shape[1]

        if base_select == 'LR':
            base_model = LogisticRegressionModel(input_dim).to(device)
            base_model.train_model(X_train_enc, Y_train)
        elif base_select == 'DNN':
            base_model = DeepNeuralNetwork(input_dim).to(device)
            base_model.train_model(X_train_enc, Y_train)
        else:
            raise ValueError(f"Unsupported base model type: {base_select}")

        # Retraining baseline
        if base_select == 'LR':
            retrainer = LogisticRegressionModel(input_dim).to(device)
            start_time = time.time()
            retrainer.train_model(X_mod_enc, Y_mod)
            end_time = time.time()
            runtime['Retrainer'] = end_time - start_time
        elif base_select == 'DNN':
            retrainer = DeepNeuralNetwork(input_dim).to(device)
            start_time = time.time()
            retrainer.train_model(X_mod_enc, Y_mod)
            end_time = time.time()
            runtime['Retrainer'] = end_time - start_time

        static_models['Retrainer'] = retrainer

        # SISA baseline (optional)
        if model == 'all' or model == 'SISA':
            if base_select == "LR":
                sisa = SISA_LR(n_shards, input_dim)
                sisa.training(X_train_enc, Y_train)
                start_time = time.time()
                sisa.retrain_affected_shards(X_train_enc, Y_train, changed_data)
                end_time = time.time()
                runtime['SISA'] = end_time - start_time
                static_models['SISA'] = sisa
            elif base_select == "DNN":
                input_dim = X_train_enc.shape[1]
                sisa_dnn = SISA_DNN(n_shards, input_dim)
                sisa_dnn.training(X_train_enc, Y_train)
                start_time = time.time()
                sisa_dnn.retrain_affected_shards(X_train_enc, Y_train, changed_data)
                end_time = time.time()
                runtime['SISA'] = end_time - start_time
                static_models['SISA'] = sisa_dnn

        # Evaluate and store static baselines
        static_performance = get_all_performance(
            static_models, retrainer,
            X_test_enc, Y_test,
            x_sample_mod_enc, y_sample_mod,
            X_proba
        )

        for static_model in static_models.keys():
            new_row = pd.DataFrame({
                "Model": [static_model],
                "Test_AUC": [static_performance[static_model]['AUC']],
                "FPR": [static_performance[static_model]['FPR']],
                "Test_Accuracy": [static_performance[static_model]['Test Accuracy']],
                "Mod_Sample_Accuracy": [static_performance[static_model]['Sample Accuracy']],
                "MIA AUC.": [static_performance[static_model]['MIA_AUC']],
                "JS Div.": [static_performance[static_model]['JS Div.']],
                "Parameter": [None],
                "Run": [run + 1],
                "Run_Time": [runtime.get(static_model, None)]
            }).dropna(axis=1, how='all')

            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                run_results = pd.concat([run_results, new_row], ignore_index=True)

        weights = calculate_class_weight(Y_mod)

        # Sweep parameters for dynamic methods
        for i in tqdm(range(num_points), desc="Running"):
            fo_tao = fo_tao_values[i]
            ft_C = ft_C_values[i]
            forgetting_rate = forgetting_rate_values[i]
            retuning_rate = retuning_rate_values[i]

            if model == 'all' or model == 'First_Order':
                fo = First_Order(base_model, base_select, fo_tao, weights)
                start_time = time.time()
                fo_model = fo.training(X_mod_enc, Y_mod, x_sample_enc, y_sample, x_sample_mod_enc, y_sample_mod)
                end_time = time.time()
                runtime['First_Order'] = end_time - start_time
                dynamic_models['First_Order'] = fo_model

            if model == 'all' or model == 'Fine_Tuning':
                ft = Fine_Tuning(base_model, base_select, ft_C, weights)
                start_time = time.time()
                ft_model = ft.training(X_mod_enc, Y_mod, x_sample_enc, y_sample, x_sample_mod_enc, y_sample_mod)
                end_time = time.time()
                runtime['Fine_Tuning'] = end_time - start_time
                dynamic_models['Fine_Tuning'] = ft_model

            if model == 'all' or model == 'BFR':
                bfr = BFR(base_model, base_select, forgetting_rate, retuning_rate, weights)
                start_time = time.time()
                bfr_model = bfr.training(x_sample_enc, y_sample, x_sample_mod_enc, y_sample_mod, X_rt2_t, Y_rt2_t)
                end_time = time.time()
                runtime['BFR'] = end_time - start_time + sampling_time
                dynamic_models['BFR'] = bfr_model

            performance = get_all_performance(
                dynamic_models, retrainer,
                X_test_enc, Y_test,
                x_sample_mod_enc, y_sample_mod,
                X_proba
            )

            # Store per-parameter evaluation results
            for ppc_model in ppc_models:
                if ppc_model in performance:
                    new_row = pd.DataFrame({
                        "Model": [ppc_model],
                        "Test_AUC": [performance[ppc_model]['AUC']],
                        "FPR": [performance[ppc_model]['FPR']],
                        "Test_Accuracy": [performance[ppc_model]['Test Accuracy']],
                        "Mod_Sample_Accuracy": [performance[ppc_model]['Sample Accuracy']],
                        "MIA AUC.": [performance[ppc_model]['MIA_AUC']],
                        "JS Div.": [performance[ppc_model]['JS Div.']],
                        "Parameter": [fo_tao if ppc_model == "First_Order" else ft_C if ppc_model == "Fine_Tuning" else forgetting_rate],
                        "Run": [run + 1],
                        "Run_Time": [runtime.get(ppc_model, None)]
                    }).dropna(axis=1, how='all')

                    with warnings.catch_warnings():
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        run_results = pd.concat([run_results, new_row], ignore_index=True)
                else:
                    print(f"Warning: No performance data for {ppc_model} model")

        cumulative_results = pd.concat([cumulative_results, run_results], ignore_index=True)

    # Save aggregated results
    #cumulative_results.to_csv(result_file, index=False)
    print(f"All results saved to {result_file}")


def test_model_performance(data_engineering_fn, dataset_path, modification_type, sample_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Prepare data
    X_train_enc, Y_train, X_test_enc, Y_test, X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod, x_sample_enc, y_sample, indices_sample, X_proba, y_proba = data_engineering_fn(
        dataset_path, modification_type, sample_num
    )

    X_train_enc = torch.tensor(X_train_enc.toarray(), dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.float32).to(device)
    X_test_enc = torch.tensor(X_test_enc.toarray(), dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test.to_numpy(), dtype=torch.float32).to(device)
    X_mod_enc = torch.tensor(X_mod_enc.toarray(), dtype=torch.float32).to(device)
    Y_mod = torch.tensor(Y_mod.to_numpy(), dtype=torch.float32).to(device)
    x_sample_mod_enc = torch.tensor(x_sample_mod_enc.toarray(), dtype=torch.float32).to(device)
    y_sample_mod = torch.tensor(y_sample_mod.to_numpy(), dtype=torch.float32).to(device)
    X_proba = torch.tensor(X_proba.toarray(), dtype=torch.float32).to(device)

    # Train an initial model
    input_dim = X_train_enc.shape[1]
    model = DeepNeuralNetwork(input_dim)
    model.train_model(X_train_enc, Y_train)

    # Train a retrained model on the modified dataset
    retrained_model = DeepNeuralNetwork(input_dim)
    retrained_model.train_model(X_mod_enc, Y_mod)

    # Evaluate
    evaluation_results = model.evaluate_model(
        retrained_model,
        X_test_enc, Y_test,
        x_sample_mod_enc, y_sample_mod,
        X_proba
    )

    print("Model Evaluation Results:")
    for key, value in evaluation_results.items():
        print(f"{key}: {value}")

    return evaluation_results

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Configuration ---
    # Dataset paths
    UCI_DATASET_PATH = "/root/5_MLForAC/dataSet/cleaned_uci-2.0.csv"
    KAGGLE_DATASET_PATH = "/root/5_MLForAC/dataSet/train.csv"

    # Dataset selection: (path, name)
    CURRENT_DATASET = (KAGGLE_DATASET_PATH, "kaggle")

    # Experiment settings
    MODIFICATION_TYPE = "mod_labels_only"   # Options: 'mod_features_and_labels', 'mod_labels_only'
    MODEL_ARCHITECTURE = "LR"               # Options: 'LR', 'DNN'

    NUM_RUNS = 10
    NUM_POINTS = 900                        # Grid resolution
    SAMPLE_NUM = 80                         # Number of PUTs (Policy Update Tasks)
    N_SHARDS = 3                            # For SISA

    # Target method selection
    TARGET_METHOD = "all"                   # Options: 'all', 'BFR', 'First_Order', 'Fine_Tuning', 'SISA'

    # --- Execution ---
    dataset_path, dataset_name = CURRENT_DATASET

    run_comprehensive_evaluation(
        num_runs=NUM_RUNS,
        num_points=NUM_POINTS,
        sample_num=SAMPLE_NUM,
        n_shards=N_SHARDS,
        dataset_path=dataset_path,
        modification_type=MODIFICATION_TYPE,
        data_type=dataset_name,              # maps to the original signature
        base_select=MODEL_ARCHITECTURE,      # maps to the original signature
        model=TARGET_METHOD                  # maps to the original signature
    )

    # Optional: quick diagnostic (kept disabled by default)
    # results = test_model_performance(data_engineering, dataset_path, MODIFICATION_TYPE, SAMPLE_NUM)
    # print(results)

