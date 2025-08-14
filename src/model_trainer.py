"""
model_trainer.py

This script contains functions for training and hyperparameter tuning
various classification models using Optuna for Bayesian optimization.
"""

import os
import joblib
import json
import numpy as np
import optuna
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def train_and_tune_model(model_name, X_train, y_train, X_test, y_test, models_path, results_path, random_state=42):
    """
    Trains and tunes a specified classification model using Optuna for hyperparameter optimization.
    Saves the best model, tuning results, and classification report.

    Args:
        model_name (str): Name of the model (e.g., 'LinearSVM', 'RBF_SVM', 'LogisticRegression', 'XGBoost', 'MLPClassifier').
        X_train (np.array): Training features (fully preprocessed and feature selected).
        y_train (np.array): Training labels.
        X_test (np.array): Testing features.
        y_test (np.array): Testing labels.
        models_path (str): Directory path to save trained models.
        results_path (str): Directory path to save JSON results.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (best_model, classification_report_dict)
    """
    print(f"\n--- Training and Tuning {model_name} ---")

    def objective(trial):
        """Optuna objective function for hyperparameter optimization."""
        if model_name == 'LinearSVM':
            C = trial.suggest_loguniform('C', 1e-4, 1e4)
            model = SVC(kernel='linear', C=C, random_state=random_state, probability=True)
        elif model_name == 'RBF_SVM':
            C = trial.suggest_loguniform('C', 1e-1, 1e4)
            gamma = trial.suggest_loguniform('gamma', 1e-4, 1e2)
            model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=random_state, probability=True)
        elif model_name == 'LogisticRegression':
            C = trial.suggest_loguniform('C', 1e-4, 1e4)
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
            # 'liblinear' supports L1/L2, 'lbfgs' supports L2. Let's use L2 for simplicity with 'lbfgs'.
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2']) if solver == 'liblinear' else 'l2'
            model = LogisticRegression(C=C, solver=solver, penalty=penalty, random_state=random_state, max_iter=1000)
        elif model_name == 'XGBoost':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False, # Suppress warning
                'seed': random_state,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_loguniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0), # L2 regularization
                'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0), # L1 regularization
            }
            model = XGBClassifier(**params)
        elif model_name == 'MLPClassifier':
            hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)])
            activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
            solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
            alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2) # L2 regularization
            learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-2)
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                  solver=solver, alpha=alpha, learning_rate_init=learning_rate_init,
                                  random_state=random_state, max_iter=1000) # Increased max_iter

        else:
            raise ValueError("Unsupported model_name provided to Optuna objective.")

        # Use StratifiedKFold for cross-validation to maintain class balance
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=kf, scoring='accuracy').mean()
        return score

    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize', study_name=f'{model_name}_optimization')
    study.optimize(objective, n_trials=50) # Increased trials to 50 for better exploration

    best_params = study.best_params
    best_cv_score = study.best_value

    print(f"\nBest parameters for {model_name}: {best_params}")
    print(f"Best cross-validation accuracy for {model_name}: {best_cv_score:.4f}")

    # Retrain the best model on the full training data
    if model_name == 'LinearSVM':
        best_model = SVC(kernel='linear', C=best_params['C'], random_state=random_state, probability=True)
    elif model_name == 'RBF_SVM':
        best_model = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], random_state=random_state, probability=True)
    elif model_name == 'LogisticRegression':
        solver = best_params['solver']
        penalty = best_params['penalty'] if solver == 'liblinear' else 'l2' # Ensure penalty matches solver
        best_model = LogisticRegression(C=best_params['C'], solver=solver, penalty=penalty, random_state=random_state, max_iter=1000)
    elif model_name == 'XGBoost':
        best_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                                   seed=random_state, **best_params)
    elif model_name == 'MLPClassifier':
        best_model = MLPClassifier(random_state=random_state, max_iter=1000, **best_params)
    else:
        raise ValueError("Unsupported model_name for best model instantiation.")

    best_model.fit(X_train, y_train)

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy for {model_name}: {accuracy:.4f}")
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

    # Save the best model
    model_filename = os.path.join(models_path, f'{model_name.lower().replace(" ", "_")}_model_v1.joblib')
    joblib.dump(best_model, model_filename)
    print(f"Best {model_name} model saved to: {model_filename}")

    # Convert Optuna trials DataFrame to a JSON serializable format
    trials_df = study.trials_dataframe()
    
    # ADDITION v15: Robustly convert all datetime and timedelta columns to string
    for col in trials_df.columns:
        if pd.api.types.is_datetime64_any_dtype(trials_df[col]):
            trials_df[col] = trials_df[col].astype(str) # Convert Timestamps
        elif pd.api.types.is_timedelta64_dtype(trials_df[col]):
            trials_df[col] = trials_df[col].astype(str) # Convert Timedeltas

    # Save tuning results and classification report as JSON
    tuning_results = {
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'study_trials_dataframe': trials_df.to_dict('records')
    }

    with open(os.path.join(results_path, f'tuning_results_{model_name.lower().replace(" ", "_")}_v1.json'), 'w') as f:
        json.dump(tuning_results, f, indent=4)

    with open(os.path.join(results_path, f'classification_report_{model_name.lower().replace(" ", "_")}_v1.json'), 'w') as f:
        json.dump(report, f, indent=4)

    print(f"{model_name} tuning results and classification report saved.")

    return best_model, report
