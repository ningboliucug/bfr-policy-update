import pandas as pd
import numpy as np
import torch
import time
import copy
import scipy.sparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Import project dependencies (assumed available in the current environment)
from base import DeepNeuralNetwork, LogisticRegressionModel
from unlearners import BFR
import puts_factory as puts

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

    # (2) Count classes (ensure at least two bins)
    class_counts = np.bincount(y_np, minlength=2)
    total_count = len(y_np)

    # (3) Compute weights (equivalent to the original logic)
    weights = total_count / class_counts.astype(float)
    class_weight = weights[1] / (weights[0] + weights[1])

    return torch.tensor([class_weight], dtype=torch.float32)

def balanced_sampling(X_mod, Y_mod, x_sample_mod, y_sample_mod):
    # Compute label ratios in the modified training set and in the modified sample set
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
      - Randomly select `sample_num` samples and apply the chosen modification type
      - Return the modified dataset and the affected sample pairs
      - Outputs are one-hot encoded using the provided encoder

    Returns:
      X_mod_enc, Y_mod,
      x_sample_mod_enc, y_sample_mod,
      x_sample_enc, y_sample,
      indices_sample
    """
    # Apply the selected PUT operator (same logic as the original implementation)
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


def run_ablation_study(num_runs, num_points, sample_num,
                       dataset_path, modification_type,
                       data_type, base_select='LR'):
    """
    Ablation study: compare Full BFR vs. w/o Forgetting vs. w/o Remembering.

    Workflow:
      - Pre-generate scenarios
      - Use the first `num_runs` scenarios for grid search (tuning)
      - Use the remaining scenarios for evaluation and report mean results
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ablation Study] device = {device}")
    print(f"[Settings] Base: {base_select}, Mod: {modification_type}, Data: {data_type}")

    # ==========================
    # 1. Data Preparation
    # ==========================
    print("Step 1: Loading and preparing data...")
    data = pd.read_csv(dataset_path)
    Y = data['ACTION']
    X = data.drop('ACTION', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=40
    )
    X_train, Y_train = X_train.reset_index(drop=True), Y_train.reset_index(drop=True)
    X_test, Y_test = X_test.reset_index(drop=True), Y_test.reset_index(drop=True)

    one_hot_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    one_hot_encoder = one_hot_enc.fit(X)

    X_train_enc = one_hot_encoder.transform(X_train)
    X_test_enc = one_hot_encoder.transform(X_test)

    X_train_enc_t = torch.tensor(X_train_enc.toarray(), dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train.to_numpy(), dtype=torch.float32).to(device)
    X_test_enc_t = torch.tensor(X_test_enc.toarray(), dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test.to_numpy(), dtype=torch.float32).to(device)

    # ==========================
    # 2. Base Model Training
    # ==========================
    print("Step 2: Training base model (theta*)...")
    input_dim = X_train_enc_t.shape[1]
    if base_select == 'LR':
        base_model = LogisticRegressionModel(input_dim).to(device)
    else:
        base_model = DeepNeuralNetwork(input_dim).to(device)

    base_model.train_model(X_train_enc_t, Y_train_t)
    base_state = copy.deepcopy(base_model.state_dict())

    # ==========================
    # 3. Scenario Generation
    # ==========================
    total_scenarios = num_runs + 100
    print(f"Step 3: Pre-generating {total_scenarios} scenarios "
          f"({num_runs} for tuning, remaining for evaluation)...")

    scenarios = []
    for j in range(total_scenarios):
        (
            X_mod_enc,
            Y_mod,
            x_sample_mod_enc,
            y_sample_mod,
            x_sample_enc,
            y_sample,
            indices_sample
        ) = sample_put_from_train(
            X_train, Y_train, one_hot_encoder, modification_type, sample_num
        )

        sampling_start = time.time()
        X_rt2, Y_rt2 = balanced_sampling(
            X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod
        )
        sampling_end = time.time()
        sampling_time = sampling_end - sampling_start

        scen_data = {
            "Y_mod": torch.tensor(Y_mod.to_numpy(), dtype=torch.float32).to(device),
            "x_sample_mod": torch.tensor(
                x_sample_mod_enc.toarray(), dtype=torch.float32
            ).to(device),
            "y_sample_mod": torch.tensor(
                y_sample_mod.to_numpy(), dtype=torch.float32
            ).to(device),
            "x_sample": torch.tensor(
                x_sample_enc.toarray(), dtype=torch.float32
            ).to(device),
            "y_sample": torch.tensor(
                y_sample.to_numpy(), dtype=torch.float32
            ).to(device),
            "X_rt2": torch.tensor(
                X_rt2.toarray(), dtype=torch.float32
            ).to(device),
            "Y_rt2": torch.tensor(
                Y_rt2.values, dtype=torch.float32
            ).to(device),
            "weights": calculate_class_weight(
                torch.tensor(Y_mod.to_numpy(), dtype=torch.float32).to(device)
            ),
            "sampling_time": sampling_time,
        }
        scenarios.append(scen_data)

    tuning_scenarios = scenarios[:num_runs]     # for grid search
    eval_scenarios = scenarios[num_runs:]       # for evaluation

    # ==========================
    # 4. Hyperparameter Grid
    # ==========================
    if base_select == 'LR':
        if data_type == 'kaggle':
            fr_max = 4.16
            rr_max = 1.73
        elif data_type == 'uci':
            fr_max = 4.53
            rr_max = 2.39
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")
    else:  # DNN
        if data_type == 'kaggle':
            fr_max = 0.04
            rr_max = 0.014
        elif data_type == 'uci':
            fr_max = 0.08
            rr_max = 0.024 
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

    grid_fr = np.linspace(0, fr_max, num_points)
    grid_rr = np.linspace(0, rr_max, num_points)

    variants = [
        "w/o Forgetting",   # alpha = 0, search beta
        "w/o Remembering",  # beta = 0, search alpha
        "Full BFR"          # search both
    ]

    ablation_results = []

    # ==========================
    # 5. Ablation Execution
    # ==========================
    print("Step 4: Running independent optimization for each variant...")

    for variant in variants:
        print(f"--- Running Variant: {variant} ---")

        if variant == "w/o Forgetting":
            search_alphas = [0.0]
            search_betas = grid_rr
        elif variant == "w/o Remembering":
            search_alphas = grid_fr
            search_betas = [0.0]
        else:  # Full BFR
            search_alphas = grid_fr
            search_betas = grid_rr

        best_mdo = float("inf")
        best_alpha = None
        best_beta = None
        best_val_auc = None
        best_val_acc = None
        best_val_runtime = None

        # Grid search on tuning_scenarios only
        for fr in search_alphas:
            for rr in search_betas:
                temp_mdo_list = []
                temp_auc_list = []
                temp_acc_list = []
                temp_time_list = []

                for scen in tuning_scenarios:
                    # Reset model to theta*
                    if base_select == 'LR':
                        model_j = LogisticRegressionModel(input_dim).to(device)
                    else:
                        model_j = DeepNeuralNetwork(input_dim).to(device)
                    model_j.load_state_dict(base_state)

                    bfr = BFR(
                        model_j,
                        base_select,
                        fr,                  # forgetting_rate
                        rr,                  # retuning_rate
                        scen["weights"]
                    )

                    start_t = time.time()
                    bfr_model = bfr.training(
                        scen["x_sample"],
                        scen["y_sample"],
                        scen["x_sample_mod"],
                        scen["y_sample_mod"],
                        scen["X_rt2"],
                        scen["Y_rt2"]
                    )
                    end_t = time.time()
                    run_time = end_t - start_t + scen["sampling_time"]

                    perf = compute_bfr_metrics(
                        {"BFR": bfr_model},
                        X_test_enc_t,
                        Y_test_t,
                        scen["x_sample_mod"],
                        scen["y_sample_mod"]
                    )
                    auc = perf["BFR"]["AUC"]
                    acc = perf["BFR"]["Sample Accuracy"]

                    # Objective used in this script (kept unchanged)
                    mdo = float(np.sqrt((acc - auc) ** 2))

                    temp_mdo_list.append(mdo)
                    temp_auc_list.append(auc)
                    temp_acc_list.append(acc)
                    temp_time_list.append(run_time)

                avg_mdo = float(np.mean(temp_mdo_list))

                if avg_mdo < best_mdo:
                    best_mdo = avg_mdo
                    best_alpha = fr
                    best_beta = rr
                    #best_val_auc = float(np.mean(temp_auc_list))
                    #best_val_acc = float(np.mean(temp_acc_list))
                    #best_val_runtime = float(np.mean(temp_time_list))

        # Evaluate with the best (alpha, beta) on eval_scenarios and report the mean
        test_auc_list = []
        test_acc_list = []
        test_mdo_list = []
        test_time_list = []

        for eval_scen in eval_scenarios:
            if base_select == 'LR':
                model_eval = LogisticRegressionModel(input_dim).to(device)
            else:
                model_eval = DeepNeuralNetwork(input_dim).to(device)
            model_eval.load_state_dict(base_state)

            bfr_eval = BFR(
                model_eval,
                base_select,
                best_alpha,
                best_beta,
                eval_scen["weights"]
            )

            start_t = time.time()
            bfr_eval_model = bfr_eval.training(
                eval_scen["x_sample"],
                eval_scen["y_sample"],
                eval_scen["x_sample_mod"],
                eval_scen["y_sample_mod"],
                eval_scen["X_rt2"],
                eval_scen["Y_rt2"]
            )
            end_t = time.time()
            test_run_time = end_t - start_t + eval_scen["sampling_time"]

            perf_eval = compute_bfr_metrics(
                {"BFR": bfr_eval_model},
                X_test_enc_t,
                Y_test_t,
                eval_scen["x_sample_mod"],
                eval_scen["y_sample_mod"]
            )
            test_auc = float(perf_eval["BFR"]["AUC"])
            test_acc = float(perf_eval["BFR"]["Sample Accuracy"])
            test_mdo = float(np.sqrt((1.0 - test_acc) ** 2 + (1.0 - test_auc) ** 2))

            test_auc_list.append(test_auc)
            test_acc_list.append(test_acc)
            test_mdo_list.append(test_mdo)
            test_time_list.append(test_run_time)

        mean_test_auc = float(np.mean(test_auc_list))
        mean_test_acc = float(np.mean(test_acc_list))
        mean_test_mdo = float(np.mean(test_mdo_list))
        mean_test_time = float(np.mean(test_time_list))

        print(
            f"   -> Best for {variant}: "
            f"Val MDO={best_mdo:.4f}, "
            f"Mean Test MDO={mean_test_mdo:.4f}"
        )

        ablation_results.append(
            {
                "Variant": variant,
                "Best_Alpha": best_alpha,
                "Best_Beta": best_beta,
                "Test_AUC": mean_test_auc,
                "Test_Accuracy": mean_test_acc,
                "Test_MDO": mean_test_mdo,
                "Test_Runtime (s)": mean_test_time,
            }
        )

    # ==========================
    # 6. Output
    # ==========================
    df_ablation = pd.DataFrame(ablation_results)
    
    print("\n" + "=" * 50)
    print("FINAL ABLATION STUDY RESULTS")
    print("=" * 50)
    print(df_ablation.to_string(index=False))
    print("=" * 50)

    return df_ablation


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
    MODIFICATION_TYPE = "mod_features_and_labels"  # Options: 'mod_features_and_labels', 'mod_labels_only'
    NUM_RUNS = 10                                  # Recommended: >= 5 for stable averages
    NUM_POINTS = 15                                # Grid resolution
    SAMPLE_NUM = 40                                # PUT size
    BASE_MODEL = "LR"                              # Options: 'LR', 'DNN'

    # --- Execution ---
    dataset_path, dataset_name = CURRENT_DATASET

    df = run_ablation_study(
        num_runs=NUM_RUNS,
        num_points=NUM_POINTS,
        sample_num=SAMPLE_NUM,
        dataset_path=dataset_path,
        modification_type=MODIFICATION_TYPE,
        data_type=dataset_name,
        base_select=BASE_MODEL
    )
