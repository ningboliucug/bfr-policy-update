"""
unlearners.py

This module implements a suite of machine unlearning algorithms including exact methods (Retraining, SISA)
and approximate gradient-based methods (First-Order, Fine-Tuning, BFR).

Design Pattern:
- Strategy Pattern: Each unlearning method follows a consistent training interface.
- Mixins: Shared evaluation and gradient logic are encapsulated in mixins for modularity.
- Wrapper/Delegation: Unlearners wrap base models and delegate evaluation to them.
"""

from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from copy import deepcopy
from scipy.sparse import vstack, csr_matrix, lil_matrix
import time
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from scipy.stats import laplace
from base import DeepNeuralNetwork, LogisticRegressionModel
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_class_weight(Y_train):
    class_counts = np.bincount(Y_train.cpu().numpy().astype(int))
    total_count = len(Y_train)
    weights = total_count / class_counts
    class_weight = weights[1] / (weights[0] + weights[1])  # Binary classification with labels {0, 1}
    return torch.tensor([class_weight], dtype=torch.float32)

class Retrainer:
    def __init__(self, base_model):
        self.model = deepcopy(base_model)
        
    def training(self, X_train_mod, Y_train_mod):
        self.model.fit(X_train_mod, Y_train_mod)
        return self.model

class First_Order:
    def __init__(self, base_model, base_select, first_order_tao, weights):
        self.tao = first_order_tao
        self.model = deepcopy(base_model)
        self.base_select = base_select
        self.weights = weights
    
    def training(self, X_train_mod, Y_train_mod, x_sample, y_sample, x_sample_mod, y_sample_mod):
        if self.base_select in ['LR', 'DNN']:            
            self.model.to(device)
            x_sample, y_sample = x_sample.to(device), y_sample.to(device)
            x_sample_mod, y_sample_mod = x_sample_mod.to(device), y_sample_mod.to(device)
                        
            # Compute gradients on original vs. modified samples
            gradient_ori = compute_gradient_sum(self.model, x_sample, y_sample, self.weights.to(device))
            gradient_mod = compute_gradient_sum(self.model, x_sample_mod, y_sample_mod, self.weights.to(device))
            
            # Apply first-order parameter update
            delta_theta = [-self.tao * (g_mod - g_ori) for g_ori, g_mod in zip(gradient_ori, gradient_mod)]
            self.model = update_model_parameters(self.model, delta_theta)
            return self.model
        else:
            raise ValueError("Unsupported base model type. Please choose either 'LR' or 'DNN'.")

class Fine_Tuning:
    def __init__(self, base_model, base_select, ft, weights):
        self.model = deepcopy(base_model)
        self.ft = ft
        self.base_select = base_select
        self.weights = weights

    def training(self, X_mod, Y_mod, x_sample, y_sample, x_sample_mod, y_sample_mod, num_epochs=10):
        # Early-stop loss thresholds (dataset/model-specific)
        if self.base_select == "LR":
            loss_min = 0.027
        elif self.base_select == "DNN":
            loss_min = 0.025
        if self.base_select in ["LR", "DNN"]:
            x_sample_mod, y_sample_mod = x_sample_mod.to(device), y_sample_mod.to(device)
            self.model.to(device)
            self.model.train()
            criterion = nn.BCEWithLogitsLoss(pos_weight = self.weights.to(device))
            optimizer = optim.Adam(self.model.parameters(), lr=self.ft, weight_decay=1e-6)
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = self.model(x_sample_mod).squeeze()
                loss = criterion(outputs, y_sample_mod)
                loss.backward()
                optimizer.step()
                avg_loss = loss.item()
                # Stop once reaching the target loss floor
                if avg_loss < loss_min:
                    break
            return self.model
        else:
            raise ValueError("Unsupported base model type. Please choose either 'LR' or 'DNN'.")

class SISA_LR:
    def __init__(self, n_shards, input_dim, lr=0.009, epochs=5, batch_size=128, weight_decay=1e-4, patience=10):
        self.n_shards = n_shards
        self.models_dict = {}
        self.splits = []
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.input_dim = input_dim

    def _train_shard(self, X_train, y_train, shard_id):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, y_train = X_train.to(device), y_train.to(device)
        model = LogisticRegressionModel(self.input_dim).to(device)
        model.train_model(X_train, y_train, epochs=self.epochs, lr=self.lr, batch_size=self.batch_size, weight_decay=self.weight_decay)
        self.models_dict[shard_id] = model

    def training(self, X, y):
        self.splits = self._stratified_split_indices(y)
        self.models_dict = {}
        for i, indices in enumerate(self.splits):
            X_split_train = X[indices].to_dense().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            y_split_train = y[indices].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self._train_shard(X_split_train, y_split_train, i)

    def _stratified_split_indices(self, y):
        class_0_indices = (y == 0).nonzero(as_tuple=True)[0].cpu().numpy()  # Indices for class 0
        class_1_indices = (y == 1).nonzero(as_tuple=True)[0].cpu().numpy()  # Indices for class 1

        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)

        # Split indices into shards per class
        class_0_splits = np.array_split(class_0_indices, self.n_shards)
        class_1_splits = np.array_split(class_1_indices, self.n_shards)

        # Merge per-shard class indices and reshuffle
        shards = [np.concatenate([class_0_split, class_1_split]) for class_0_split, class_1_split in zip(class_0_splits, class_1_splits)]
        for shard in shards:
            np.random.shuffle(shard)
        return shards

    def find_and_update_affected_shards(self, X, y, changed_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        affected_shards = set()
        X = X.to(device)
        y = y.to(device)
        for idx, (new_x, new_y) in changed_data.items():
            new_x_tensor = torch.tensor(new_x, dtype=torch.float32).to(device)
            new_y_tensor = torch.tensor(new_y, dtype=torch.float32).to(device)
            for shard_id, split_indices in enumerate(self.splits):
                if idx in split_indices:
                    affected_shards.add(shard_id)
                    X[idx] = new_x_tensor
                    y[idx] = new_y_tensor
                    break
        return affected_shards, X

    def retrain_affected_shards(self, X, y, changed_data):
        affected_shards, X = self.find_and_update_affected_shards(X, y, changed_data)
        for shard_id in affected_shards:
            shard_indices = self.splits[shard_id]
            X_shard = X[shard_indices]
            y_shard = y[shard_indices]
            self._train_shard(X_shard, y_shard, shard_id)

    def _aggregate_predict_mean(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        sum_of_probabilities = None
        for model in self.models_dict.values():
            probabilities = torch.sigmoid(model(X)).to('cpu').numpy().flatten()
            if sum_of_probabilities is None:
                sum_of_probabilities = probabilities
            else:
                sum_of_probabilities += probabilities
        mean_probabilities = sum_of_probabilities / len(self.models_dict)
        return mean_probabilities

    def _aggregate_predict_proba_most_confidence(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        
        most_confident_predictions = np.zeros((X.shape[0], 1))  # Binary classification
        max_confidences = np.full(X.shape[0], -np.inf)

        for model in self.models_dict.values():
            logits = model(X)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            confidences = np.abs(probabilities[:, 0] - 0.5)  # Distance from 0.5 as confidence

            for i in range(X.shape[0]):
                if confidences[i] > max_confidences[i]:
                    max_confidences[i] = confidences[i]
                    most_confident_predictions[i] = probabilities[i]

        return most_confident_predictions

    def predict(self, X):
        return self._aggregate_predict_mean(X)
    
    def predict_proba(self, X):
        return self._aggregate_predict_proba_most_confidence(X)

    def evaluate_model(self, retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move all shard models to device
        for model in self.models_dict.values():
            model.to(device)
        retrainer.model.to(device)
        
        # Switch to eval mode
        for model in self.models_dict.values():
            model.eval()
        retrainer.model.eval()
        
        with torch.no_grad():  
            # Aggregate probabilities for test set
            test_probas = self._aggregate_predict_proba_most_confidence(X_test.to(device))
            test_probas_cpu = test_probas.squeeze()
            
            # Select threshold via ROC/Youden's J statistic
            Y_test_cpu = Y_test.cpu().numpy()
            fpr, tpr, thresholds = roc_curve(Y_test_cpu, test_probas_cpu)
            youden_index = tpr - fpr
            best_threshold_index_youden = np.argmax(youden_index)
            best_threshold_youden = thresholds[best_threshold_index_youden]
            
            # Predict using the selected threshold
            test_predictions = (test_probas_cpu > best_threshold_youden).astype(int)
            
            # Compute metrics
            auc = roc_auc_score(Y_test_cpu, test_probas_cpu)
            test_accuracy = accuracy_score(Y_test_cpu, test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(Y_test_cpu, test_predictions, average='binary')
            
            tn, fp, fn, tp = confusion_matrix(Y_test_cpu, test_predictions).ravel()
            fpr_final = fp / (fp + tn)
            tpr_final = tp / (tp + fn)
            
            # Sample accuracy on modified samples
            sample_mod_outputs = self._aggregate_predict_proba_most_confidence(X_sample_mod.to(device))
            sample_mod_probas = sample_mod_outputs.squeeze()
            sample_mod_predictions = (sample_mod_probas > best_threshold_youden).astype(int)
            sample_accuracy = accuracy_score(Y_sample_mod.cpu().numpy(), sample_mod_predictions)
            
            # Aggregate probabilities for X_proba (for downstream analysis)
            proba_outputs = self._aggregate_predict_proba_most_confidence(X_proba.to(device)).squeeze()
            
            # Jensen–Shannon divergence vs. retrainer
            js_div = self.calculate_js_divergence_combined(retrainer, X_test, X_sample_mod)
            
            # Membership inference attack (AUC)
            mia_accuracy, mia_roc_auc = self.mia_attack(test_probas_cpu, sample_mod_probas)
            
            return {
                'AUC': auc,
                'Test Accuracy': test_accuracy,
                'FPR': fpr_final,
                'TPR': tpr_final,
                'F1': f1,
                'Sample Accuracy': sample_accuracy,
                'JS Div.': js_div,
                'MIA_AUC': mia_roc_auc
            }

    def calculate_js_divergence_combined(self, retrainer, X, X_sample):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in self.models_dict.values():
            model.to(device)
        retrainer.model.to(device)
        
        for model in self.models_dict.values():
            model.eval()
        retrainer.model.eval()
        
        X_combined_tensor = torch.cat([X, X_sample], dim=0).to(device)
        
        with torch.no_grad():
            retrainer_probas = torch.sigmoid(retrainer.model(X_combined_tensor)).cpu().numpy().flatten()
            unlearner_probas = self._aggregate_predict_proba_most_confidence(X_combined_tensor.to(device)).squeeze()
            js_div = jensenshannon(retrainer_probas, unlearner_probas, base=2)
            
        return js_div

    def mia_attack(self, train_probs, test_probs):
        # Train a simple LR attack model on predicted probabilities
        mia_X = np.concatenate((train_probs, test_probs)).reshape(-1, 1)
        mia_y = np.concatenate((np.ones_like(train_probs), np.zeros_like(test_probs)))

        attack_model = SklearnLogisticRegression(solver='lbfgs', class_weight='balanced')
        attack_model.fit(mia_X, mia_y)

        mia_preds = attack_model.predict(mia_X)
        accuracy = accuracy_score(mia_y, mia_preds)
        roc_auc = roc_auc_score(mia_y, attack_model.predict_proba(mia_X)[:, 1])

        return accuracy, roc_auc

class SISA_DNN:    
    def __init__(self, n_shards, input_dim, lr=0.001, epochs=5, batch_size=128, weight_decay=1e-4, patience=10):
        self.n_shards = n_shards
        self.models_dict = {}
        self.splits = []
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.input_dim = input_dim

    def _train_shard(self, X_train, y_train, shard_id):
        X_train, y_train = X_train.to(device), y_train.to(device)
        model = DeepNeuralNetwork(self.input_dim).to(device)
        model.train_model(X_train, y_train)
        self.models_dict[shard_id] = model

    def training(self, X, y):
        self.splits = self._stratified_split_indices(y)
        self.models_dict = {}
        for i, indices in enumerate(self.splits):
            X_split_train = X[indices].to_dense().to(device)
            y_split_train = y[indices].to(device)
            self._train_shard(X_split_train, y_split_train, i)

    def _stratified_split_indices(self, y):
        class_0_indices = (y == 0).nonzero(as_tuple=True)[0].cpu().numpy()  # Indices for class 0
        class_1_indices = (y == 1).nonzero(as_tuple=True)[0].cpu().numpy()  # Indices for class 1

        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)

        # Split indices into shards per class
        class_0_splits = np.array_split(class_0_indices, self.n_shards)
        class_1_splits = np.array_split(class_1_indices, self.n_shards)

        # Merge per-shard class indices and reshuffle
        shards = [np.concatenate([class_0_split, class_1_split]) for class_0_split, class_1_split in zip(class_0_splits, class_1_splits)]
        for shard in shards:
            np.random.shuffle(shard)
        return shards

    def find_and_update_affected_shards(self, X, y, changed_data):
        affected_shards = set()
        X = X.to(device)
        y = y.to(device)
        for idx, (new_x, new_y) in changed_data.items():
            new_x_tensor = torch.tensor(new_x, dtype=torch.float32).to(device)
            new_y_tensor = torch.tensor(new_y, dtype=torch.float32).to(device)
            for shard_id, split_indices in enumerate(self.splits):
                if idx in split_indices:
                    affected_shards.add(shard_id)
                    X[idx] = new_x_tensor
                    y[idx] = new_y_tensor
                    break
        return affected_shards, X

    def retrain_affected_shards(self, X, y, changed_data):
        affected_shards, X = self.find_and_update_affected_shards(X, y, changed_data)
        for shard_id in affected_shards:
            shard_indices = self.splits[shard_id]
            X_shard = X[shard_indices]
            y_shard = y[shard_indices]
            self._train_shard(X_shard, y_shard, shard_id)

    def _aggregate_predict_mean(self, X):
        X = X.to(device)
        sum_of_probabilities = None
        for model in self.models_dict.values():
            probabilities = model.predict_proba(X).to('cpu').numpy()[:, 1]
            if sum_of_probabilities is None:
                sum_of_probabilities = probabilities
            else:
                sum_of_probabilities += probabilities
        mean_probabilities = sum_of_probabilities / len(self.models_dict)
        final_prediction = (mean_probabilities > 0.5).astype(int)
        return final_prediction

    def _aggregate_predict_proba_most_confidence(self, X):
        X = X.to(device)
        
        most_confident_predictions = np.zeros((X.shape[0], 1))  # Binary classification
        max_confidences = np.full(X.shape[0], -np.inf)

        for model in self.models_dict.values():
            logits = model(X)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            confidences = np.abs(probabilities[:, 0] - 0.5)

            for i in range(X.shape[0]):
                if confidences[i] > max_confidences[i]:
                    max_confidences[i] = confidences[i]
                    most_confident_predictions[i] = probabilities[i]

        return most_confident_predictions

    def predict(self, X):
        return self._aggregate_predict_mean(X)
    
    def predict_proba(self, X):
        return self._aggregate_predict_proba_most_confidence(X)

    def evaluate_model(self, retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba):
        # Move all shard models to device
        for model in self.models_dict.values():
            model.to(device)
        retrainer.model.to(device)
        
        # Switch to eval mode
        for model in self.models_dict.values():
            model.eval()
        retrainer.model.eval()
        
        with torch.no_grad():  
            test_probas = self._aggregate_predict_proba_most_confidence(X_test.to(device))
            test_probas_cpu = test_probas.squeeze()
            
            # Select threshold via ROC/Youden's J statistic
            Y_test_cpu = Y_test.cpu().numpy()
            fpr, tpr, thresholds = roc_curve(Y_test_cpu, test_probas_cpu)
            youden_index = tpr - fpr
            best_threshold_index_youden = np.argmax(youden_index)
            best_threshold_youden = thresholds[best_threshold_index_youden]
            
            test_predictions = (test_probas_cpu > best_threshold_youden).astype(int)
            
            auc = roc_auc_score(Y_test_cpu, test_probas_cpu)
            test_accuracy = accuracy_score(Y_test_cpu, test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(Y_test_cpu, test_predictions, average='binary')
            
            tn, fp, fn, tp = confusion_matrix(Y_test_cpu, test_predictions).ravel()
            fpr_final = fp / (fp + tn)
            tpr_final = tp / (tp + fn)
            
            # Sample accuracy on modified samples
            sample_mod_outputs = self._aggregate_predict_proba_most_confidence(X_sample_mod.to(device))
            sample_mod_probas = sample_mod_outputs.squeeze()
            sample_mod_predictions = (sample_mod_probas > best_threshold_youden).astype(int)
            sample_accuracy = accuracy_score(Y_sample_mod.cpu().numpy(), sample_mod_predictions)
            
            # Aggregate probabilities for X_proba (for downstream analysis)
            proba_outputs = self._aggregate_predict_proba_most_confidence(X_proba.to(device)).squeeze()
            
            # Jensen–Shannon divergence vs. retrainer
            js_div = self.calculate_js_divergence_combined(retrainer, X_test, X_sample_mod)
            
            # Membership inference attack (AUC)
            mia_accuracy, mia_roc_auc = self.mia_attack(test_probas_cpu, sample_mod_probas)
            
            return {
                'AUC': auc,
                'Test Accuracy': test_accuracy,
                'FPR': fpr_final,
                'TPR': tpr_final,
                'F1': f1,
                'Sample Accuracy': sample_accuracy,
                'JS Div.': js_div,
                'MIA_AUC': mia_roc_auc
            }

    def calculate_js_divergence_combined(self, retrainer, X, X_sample):
        for model in self.models_dict.values():
            model.to(device)
        retrainer.model.to(device)
        
        for model in self.models_dict.values():
            model.eval()
        retrainer.model.eval()
        
        X_combined_tensor = torch.cat([X, X_sample], dim=0).to(device)
        
        with torch.no_grad():
            retrainer_probas = torch.sigmoid(retrainer.model(X_combined_tensor)).cpu().numpy().flatten()
            unlearner_probas = self._aggregate_predict_proba_most_confidence(X_combined_tensor.to(device)).squeeze()
            js_div = jensenshannon(retrainer_probas, unlearner_probas, base=2)
            
        return js_div

    def mia_attack(self, train_probs, test_probs):
        mia_X = np.concatenate((train_probs, test_probs)).reshape(-1, 1)
        mia_y = np.concatenate((np.ones_like(train_probs), np.zeros_like(test_probs)))

        attack_model = SklearnLogisticRegression(solver='lbfgs', class_weight='balanced')
        attack_model.fit(mia_X, mia_y)

        mia_preds = attack_model.predict(mia_X)
        accuracy = accuracy_score(mia_y, mia_preds)
        roc_auc = roc_auc_score(mia_y, attack_model.predict_proba(mia_X)[:, 1])

        return accuracy, roc_auc

class BFR:
    def __init__(self, base_model, base_select, forgetting_rate, retuning_rate, weights, weight_decay=1e-6):
        self.fr = forgetting_rate
        self.rr = retuning_rate
        self.model = deepcopy(base_model)
        self.base_select = base_select
        self.weights = weights
        self.weight_decay = weight_decay
    
    def training(self, x_sample, y_sample, x_sample_mod, y_sample_mod, X_rt2, Y_rt2):
        if self.base_select in ['LR', 'DNN']:
            self.model.to(device)
            x_sample, y_sample = x_sample.to(device), y_sample.to(device)
            x_sample_mod, y_sample_mod = x_sample_mod.to(device), y_sample_mod.to(device)
            X_rt2, Y_rt2 = X_rt2.to(device), Y_rt2.to(device)
            # Stage 1: Forgetting (first-order update)
            gradient_ori = self.compute_gradient_sum(x_sample, y_sample, self.weights.to(device))
            gradient_mod = self.compute_gradient_sum(x_sample_mod, y_sample_mod, self.weights.to(device))
            delta_theta = [-self.fr * (g_mod - g_ori) for g_ori, g_mod in zip(gradient_ori, gradient_mod)]
            self.model = self.update_model_parameters(delta_theta)
            # Stage 2: Retuning (single-step gradient update on auxiliary set)
            self.model = self._retuning(X_rt2, Y_rt2, self.weights.to(device))
            return self.model
        else:
            raise ValueError("Unsupported base model type. Please choose either 'LR' or 'DNN'.")

    def _retuning(self, X_rt2, Y_rt2, weights):
        self.model.train()
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.rr, weight_decay = self.weight_decay)
        optimizer.zero_grad()
        outputs = self.model(X_rt2).squeeze()
        loss = criterion(outputs, Y_rt2)
        loss.backward()
        optimizer.step()
        return self.model
    
    def compute_gradient_sum(self, X, Y, class_weight):
        self.model.train()
        criterion = nn.BCEWithLogitsLoss(pos_weight = class_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        optimizer.zero_grad()
        outputs = self.model(X).squeeze()
        loss = criterion(outputs, Y)
        loss.backward()
        gradients = [param.grad.clone().detach() for param in self.model.parameters() if param.grad is not None]
        return gradients

    def update_model_parameters(self, delta_theta):
        with torch.no_grad():
            for param, delta in zip(self.model.parameters(), delta_theta):
                param.add_(delta)
        return self.model

def compute_gradient_sum(model, X, Y, class_weight):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=1)
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, Y)
    loss.backward()
    gradients = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    return gradients

def update_model_parameters(model, delta_theta):
    with torch.no_grad():
        for param, delta in zip(model.parameters(), delta_theta):
            param.add_(delta)
    return model

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
    X_ft = scipy.sparse.vstack([X_ft, x_sample_mod])  # Sparse vertical concatenation
    Y_ft = pd.concat([Y_ft, y_sample_mod])

    # Sanity check
    if X_ft.shape[0] != Y_ft.shape[0]:
        raise ValueError("The number of samples in X_ft and Y_ft must be the same.")

    return X_ft, Y_ft
