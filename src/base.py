"""
base.py

This module defines the foundational machine learning architectures and 
a unified evaluation protocol used across the benchmarking framework. 
It encapsulates model definitions, training loops, and metric computation 
logic (e.g., AUC, JS Divergence, Membership Inference Attack proxy).

Architectural Design:
    - BaseModel: Abstract base class implementing shared training and evaluation logic.
    - LogisticRegressionModel: A PyTorch implementation of Logistic Regression.
    - DeepNeuralNetwork: A Multi-Layer Perceptron (MLP) for high-dimensional feature learning.

Dependencies:
    - PyTorch 1.8+
    - Scikit-learn
    - SciPy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.spatial.distance import jensenshannon
from numpy import vstack
import random
from sklearn.metrics import roc_curve
import torch.optim as optim
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        return self.model(x)  # Forward pass through self.model
    
    def calculate_class_weight(self, Y_train):
        # Compute class weights on CPU (torch.bincount is not supported on GPU)
        class_counts = torch.bincount(Y_train.to(torch.int64).cpu())
        total_count = len(Y_train)
        weights = total_count / class_counts
        class_weight = weights[1] / (weights[0] + weights[1])  # Binary labels {0,1}
        return class_weight.to(Y_train.device)  # Move back to the original device
    
    def train_model(self, X, Y, epochs=5, lr=0.012, batch_size=128, weight_decay=1e-6): #kaggle # S1: lr=0.005; S2: lr=0.012
    #def train_model(self, X, Y, epochs=5, lr=0.009, batch_size=128, weight_decay=1e-7): #uci
        device = X.device  # Use the same device as the input tensors
        self.to(device)  # Move the model to the same device
        
        # Compute class weight for BCEWithLogitsLoss
        class_weight = self.calculate_class_weight(Y).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight], device=device))
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # DataLoader for mini-batch training
        train_dataset = CustomDataset(X.cpu().numpy(), Y.cpu().numpy())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = self(X_batch).squeeze()
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            #avg_train_loss = train_loss / len(train_loader)
            #print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}')
        
        return self
    
    def evaluate_model(self, retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        retrainer.model.to(device)
        
        self.model.eval()
        retrainer.model.eval()
        
        with torch.no_grad():
            test_outputs = self(X_test).squeeze()
            test_probas = torch.sigmoid(test_outputs)

            # Select threshold via ROC (Youden's J statistic)
            test_probas_cpu = test_probas.cpu().numpy()
            Y_test_cpu = Y_test.cpu().numpy()
            fpr, tpr, thresholds = roc_curve(Y_test_cpu, test_probas_cpu)
            youden_index = tpr - fpr
            best_threshold_index_youden = np.argmax(youden_index)
            best_threshold_youden = thresholds[best_threshold_index_youden]

            # Predict with the selected threshold
            test_predictions = (test_probas > best_threshold_youden).cpu().numpy().astype(int)
            
            auc = roc_auc_score(Y_test_cpu, test_probas_cpu)
            test_accuracy = accuracy_score(Y_test_cpu, test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(Y_test_cpu, test_predictions, average='binary')
            
            tn, fp, fn, tp = confusion_matrix(Y_test_cpu, test_predictions).ravel()
            fpr_final = fp / (fp + tn)
            tpr_final = tp / (tp + fn)
            
            # Sample accuracy on (X_sample_mod, Y_sample_mod)
            sample_mod_outputs = self(X_sample_mod).squeeze()
            sample_mod_probas = torch.sigmoid(sample_mod_outputs)
            sample_mod_predictions = (sample_mod_probas > best_threshold_youden).cpu().numpy().astype(int)
            sample_accuracy = accuracy_score(Y_sample_mod.cpu().numpy(), sample_mod_predictions)
            
            # Predicted probabilities on X_proba
            X_proba_tensor = X_proba.to(device)
            proba_outputs = torch.sigmoid(self(X_proba_tensor)).cpu().numpy()
            
            # JS divergence between retrainer and unlearner outputs
            js_div = self.calculate_js_divergence_combined(retrainer, X_test, X_sample_mod)
            
            # MIA attack AUC (attack model trained on probabilities)
            mia_accuracy, mia_roc_auc = self.mia_attack(self, test_probas_cpu.flatten(), sample_mod_probas.cpu().numpy().flatten())
            
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
    
    def evaluate_auc_sample(self, X_test, Y_test, X_sample_mod, Y_sample_mod):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            # (1) Test AUC
            test_outputs = self(X_test).squeeze()
            test_probas = torch.sigmoid(test_outputs)

            test_probas_cpu = test_probas.cpu().numpy()
            Y_test_cpu = Y_test.cpu().numpy()

            # Use ROC-derived threshold and reuse it on sample_mod
            fpr, tpr, thresholds = roc_curve(Y_test_cpu, test_probas_cpu)
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]

            auc = roc_auc_score(Y_test_cpu, test_probas_cpu)

            # (2) Sample accuracy on (X_sample_mod, Y_sample_mod)
            sample_outputs = self(X_sample_mod).squeeze()
            sample_probas = torch.sigmoid(sample_outputs)
            sample_pred = (sample_probas > best_threshold).cpu().numpy().astype(int)

            sample_accuracy = accuracy_score(
                Y_sample_mod.cpu().numpy(), sample_pred
            )

            return {
                "AUC": auc,
                "Sample Accuracy": sample_accuracy,
            }

    def calculate_js_divergence_combined(self, retrainer, X, X_sample):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        retrainer.model.to(device)
        
        self.model.eval()
        retrainer.model.eval()
        
        # Evaluate on concatenated inputs for a single JS computation
        X_combined_tensor = torch.cat([X, X_sample], dim=0).to(device)
        
        with torch.no_grad():
            retrainer_probas = torch.sigmoid(retrainer(X_combined_tensor)).cpu().numpy().flatten()
            unlearner_probas = torch.sigmoid(self(X_combined_tensor)).cpu().numpy().flatten()
            
            js_div = jensenshannon(retrainer_probas, unlearner_probas, base=2)
            
        return js_div

    def mia_attack(self, target_model, train_probs, test_probs):
        # Build attack dataset: train_probs as members, test_probs as non-members
        mia_X = np.concatenate((train_probs, test_probs)).reshape(-1, 1)
        mia_y = np.concatenate((np.ones_like(train_probs), np.zeros_like(test_probs)))

        # Train a simple logistic-regression attack model
        attack_model = SklearnLogisticRegression(solver='lbfgs', class_weight='balanced')
        attack_model.fit(mia_X, mia_y)

        # Evaluate attack model
        mia_preds = attack_model.predict(mia_X)
        accuracy = accuracy_score(mia_y, mia_preds)
        roc_auc = roc_auc_score(mia_y, attack_model.predict_proba(mia_X)[:, 1])

        return accuracy, roc_auc

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(DeepNeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def calculate_class_weight(self, Y_train):
        # Compute class weights on CPU (torch.bincount is not supported on GPU)
        class_counts = torch.bincount(Y_train.to(torch.int64).cpu())
        total_count = len(Y_train)
        weights = total_count / class_counts
        class_weight = weights[1] / (weights[0] + weights[1])  # Binary labels {0,1}
        
        return class_weight.to(Y_train.device)  # Move back to the original device
    
    #def train_model(self, X, Y, epochs=5, lr=0.001, batch_size=128, weight_decay=1e-4, patience=10): # kaggle
    def train_model(self, X, Y, epochs=5, lr=0.001, batch_size=128, weight_decay=1e-6, patience=10): # UCI
        device = X.device  # Use the same device as the input tensors
        self.model.to(device)  # Move the model to the same device
        
        # Calculate class weight for BCEWithLogitsLoss
        class_weight = self.calculate_class_weight(Y).to(device)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # DataLoader for mini-batch training
        train_dataset = CustomDataset(X.cpu().numpy(), Y.cpu().numpy())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            #print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")
            
            if avg_train_loss < 0.03:
                break
        
        return self
    
    def evaluate_model(self, retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        retrainer.model.to(device)
        
        self.model.eval()
        retrainer.model.eval()
        
        with torch.no_grad():            
            test_outputs = self.model(X_test).squeeze()
            test_probas = torch.sigmoid(test_outputs)
            
            # Select threshold via ROC (Youden's J statistic)
            test_probas_cpu = test_probas.cpu().numpy()
            Y_test_cpu = Y_test.cpu().numpy()
            fpr, tpr, thresholds = roc_curve(Y_test_cpu, test_probas_cpu)
            youden_index = tpr - fpr
            best_threshold_index_youden = np.argmax(youden_index)
            best_threshold_youden = thresholds[best_threshold_index_youden]

            # Predict with the selected threshold
            test_predictions = (test_probas > best_threshold_youden).cpu().numpy().astype(int)
            
            auc = roc_auc_score(Y_test_cpu, test_probas_cpu)
            test_accuracy = accuracy_score(Y_test_cpu, test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(Y_test_cpu, test_predictions, average='binary')
            
            tn, fp, fn, tp = confusion_matrix(Y_test_cpu, test_predictions).ravel()
            fpr_final = fp / (fp + tn)
            tpr_final = tp / (tp + fn)
            
            # Sample accuracy on (X_sample_mod, Y_sample_mod)
            sample_mod_outputs = self.model(X_sample_mod).squeeze()
            sample_mod_probas = torch.sigmoid(sample_mod_outputs)
            sample_mod_predictions = (sample_mod_probas > best_threshold_youden).cpu().numpy().astype(int)
            sample_accuracy = accuracy_score(Y_sample_mod.cpu().numpy(), sample_mod_predictions)
            
            # Predicted probabilities on X_proba
            X_proba_tensor = X_proba.to(device)
            proba_outputs = torch.sigmoid(self.model(X_proba_tensor)).cpu().numpy()
            
            # JS divergence between retrainer and unlearner outputs
            js_div = self.calculate_js_divergence_combined(retrainer, X_test, X_sample_mod)
            
            # MIA attack AUC (attack model trained on probabilities)
            #X_mia, _ = self.stratified_sample(X_test, Y_test, X_sample_mod, Y_sample_mod, factor=50)
            #train_probs = torch.sigmoid(self.model(X_mia)).cpu().numpy().flatten()
            mia_accuracy, mia_roc_auc = self.mia_attack(self.model, test_probas_cpu.flatten(), sample_mod_probas.cpu().numpy().flatten())
            
            return {
                'AUC': auc,
                'Test Accuracy': test_accuracy,
                'FPR': fpr_final,
                'TPR': tpr_final,
                'F1': f1,
                'Sample Accuracy': sample_accuracy,
                #'Probability': proba_outputs,
                'JS Div.': js_div,
                'MIA_AUC': mia_roc_auc
            }

    def evaluate_auc_sample(self, X_test, Y_test, X_sample_mod, Y_sample_mod):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            # (1) Test AUC
            test_outputs = self(X_test).squeeze()
            test_probas = torch.sigmoid(test_outputs)

            test_probas_cpu = test_probas.cpu().numpy()
            Y_test_cpu = Y_test.cpu().numpy()

            # Use ROC-derived threshold and reuse it on sample_mod
            fpr, tpr, thresholds = roc_curve(Y_test_cpu, test_probas_cpu)
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]

            auc = roc_auc_score(Y_test_cpu, test_probas_cpu)

            # (2) Sample accuracy on (X_sample_mod, Y_sample_mod)
            sample_outputs = self(X_sample_mod).squeeze()
            sample_probas = torch.sigmoid(sample_outputs)
            sample_pred = (sample_probas > best_threshold).cpu().numpy().astype(int)

            sample_accuracy = accuracy_score(
                Y_sample_mod.cpu().numpy(), sample_pred
            )

            return {
                "AUC": auc,
                "Sample Accuracy": sample_accuracy,
            }

    def calculate_js_divergence_combined(self, retrainer, X, X_sample):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        retrainer.model.to(device)
        
        self.model.eval()
        retrainer.model.eval()
        
        # Evaluate on concatenated inputs for a single JS computation
        X_combined_tensor = torch.cat([X, X_sample], dim=0).to(device)
        
        with torch.no_grad():
            retrainer_probas = torch.sigmoid(retrainer.model(X_combined_tensor)).cpu().numpy().flatten()
            unlearner_probas = torch.sigmoid(self.model(X_combined_tensor)).cpu().numpy().flatten()
            
            js_div = jensenshannon(retrainer_probas, unlearner_probas, base=2)
            
        return js_div

    def mia_attack(self, target_model, train_probs, test_probs):
        # Build attack dataset: train_probs as members, test_probs as non-members
        mia_X = np.concatenate((train_probs, test_probs)).reshape(-1, 1)
        mia_y = np.concatenate((np.ones_like(train_probs), np.zeros_like(test_probs)))

        # Train a simple logistic-regression attack model
        attack_model = SklearnLogisticRegression(solver='lbfgs', class_weight='balanced')
        attack_model.fit(mia_X, mia_y)

        # Evaluate attack model
        mia_preds = attack_model.predict(mia_X)
        accuracy = accuracy_score(mia_y, mia_preds)
        roc_auc = roc_auc_score(mia_y, attack_model.predict_proba(mia_X)[:, 1])

        return accuracy, roc_auc

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
