"""
main.py

This script orchestrates the entire machine learning classification pipeline for local execution.
It handles data loading, preprocessing, feature selection, model training (with Bayesian optimization),
evaluation, and visualization by calling functions from modular scripts in 'src/'.
It includes a granular checkpointing system to avoid retraining models unnecessarily.
"""

import os
import sys
import numpy as np
import joblib # Import joblib for loading models
import json # Import json for saving reports
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score # Import for ensemble evaluation

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import functions from the modular scripts
from data_loader import load_and_preprocess_data
from model_trainer import train_and_tune_model
from visualizer import plot_decision_boundary, plot_confusion_matrix, plot_roc_curve

def get_user_confirmation(prompt):
    """Asks the user for confirmation (y/n)."""
    while True:
        response = input(prompt).lower()
        if response in ['y', 'n']:
            return response == 'y'
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def clean_directories(paths):
    """Deletes all files within specified directories."""
    for path in paths:
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"Cleaned directory: {path}")
        else:
            print(f"Directory not found, skipping clean: {path}")


def main():
    """
    Main function to run the classification pipeline.
    """
    # Define base paths for local execution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    MODELS_PATH = os.path.join(BASE_DIR, 'models')
    RESULTS_PATH = os.path.join(BASE_DIR, 'results')
    PLOTS_PATH = os.path.join(RESULTS_PATH, 'plots')

    print("--- Starting Machine Learning Classification Pipeline ---")
    print(f"Data will be loaded from: {DATA_PATH}")
    print(f"Models will be saved to: {MODELS_PATH}")
    print(f"Results (JSON, plots) will be saved to: {RESULTS_PATH} and {PLOTS_PATH}")

    # Ask to clean models and results directories, explicitly excluding plots
    # CHANGE v20: Removed PLOTS_PATH from the clean_directories call
    if get_user_confirmation("\nDo you want to clean 'models/' and 'results/' directories before starting? (y/n): "):
        clean_directories([MODELS_PATH, RESULTS_PATH])

    # Ensure directories exist (especially plots, which might not have been cleaned)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)


    # 1. Load and preprocess data
    print("\n[Step 1/7] Loading and Preprocessing Data...")
    (X_train, X_test, y_train, y_test,
     X_train_pca, X_test_pca, plot_feature_names) = load_and_preprocess_data(DATA_PATH)

    # Define model paths for checking existence
    model_configs = {
        'LinearSVM': {'model_file': 'linear_svm_model_v1.joblib'},
        'RBF_SVM': {'model_file': 'rbf_svm_model_v1.joblib'},
        'LogisticRegression': {'model_file': 'logistic_regression_model_v1.joblib'},
        'XGBoost': {'model_file': 'xgboost_model_v1.joblib'},
        'MLPClassifier': {'model_file': 'mlpclassifier_model_v1.joblib'},
    }

    trained_models = {}

    for model_name, config in model_configs.items():
        model_path = os.path.join(MODELS_PATH, config['model_file'])
        
        # Granular checkpointing
        if os.path.exists(model_path):
            print(f"\nExisting trained {model_name} model found at {model_path}.")
            retrain = get_user_confirmation(f"Do you want to retrain {model_name}? (y/n): ")
            if not retrain:
                try:
                    trained_models[model_name] = joblib.load(model_path)
                    print(f"Loaded existing {model_name} model.")
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}. Retraining {model_name}.")
                    trained_models[model_name], _ = train_and_tune_model(
                        model_name, X_train, y_train, X_test, y_test, MODELS_PATH, RESULTS_PATH
                    )
            else:
                trained_models[model_name], _ = train_and_tune_model(
                    model_name, X_train, y_train, X_test, y_test, MODELS_PATH, RESULTS_PATH
                )
        else:
            print(f"\nNo existing {model_name} model found. Training {model_name}...")
            trained_models[model_name], _ = train_and_tune_model(
                model_name, X_train, y_train, X_test, y_test, MODELS_PATH, RESULTS_PATH
            )
            
    # Check if all models were successfully loaded or trained
    if not all(model is not None for model in trained_models.values()):
        print("Error: Not all models could be loaded or trained. Exiting.")
        sys.exit(1)


    print("\n[Step 2/7] All individual models trained or loaded.")

    # Train and Evaluate Ensemble Voting Classifier
    print("\n--- Training and Evaluating Ensemble Voting Classifier ---")
    ensemble_model_path = os.path.join(MODELS_PATH, 'ensemble_voting_classifier_v1.joblib')

    retrain_ensemble = True # Default to retraining ensemble if individual models were retrained
    if os.path.exists(ensemble_model_path):
        print(f"Existing trained Ensemble Voting Classifier found at {ensemble_model_path}.")
        retrain_ensemble = get_user_confirmation("Do you want to retrain the Ensemble Voting Classifier? (y/n): ")

    if retrain_ensemble:
        # Create a VotingClassifier from the trained individual models
        # Use 'soft' voting for probability-based averaging
        estimators = [
            ('svm_linear', trained_models['LinearSVM']),
            ('svm_rbf', trained_models['RBF_SVM']),
            ('log_reg', trained_models['LogisticRegression']),
            ('xgb', trained_models['XGBoost']),
            ('mlp', trained_models['MLPClassifier'])
        ]
        
        # Ensure that models like MLPClassifier or LogisticRegression also have probability=True
        # MLPClassifier has predict_proba by default. LogisticRegression also has it.
        # SVC needs probability=True explicitly (which we've set in model_trainer.py)
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        voting_clf.fit(X_train, y_train)

        y_pred_ensemble = voting_clf.predict(X_test)
        ensemble_report = classification_report(y_test, y_pred_ensemble, output_dict=True)
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)

        print(f"\nTest Accuracy for Ensemble Voting Classifier: {accuracy_ensemble:.4f}")
        print("Classification Report for Ensemble Voting Classifier:")
        print(classification_report(y_test, y_pred_ensemble))

        joblib.dump(voting_clf, ensemble_model_path)
        print(f"Ensemble Voting Classifier saved to: {ensemble_model_path}")

        with open(os.path.join(RESULTS_PATH, 'classification_report_ensemble_v1.json'), 'w') as f:
            json.dump(ensemble_report, f, indent=4)
        print("Ensemble Voting Classifier report saved.")
        trained_models['EnsembleVotingClassifier'] = voting_clf
    else:
        try:
            trained_models['EnsembleVotingClassifier'] = joblib.load(ensemble_model_path)
            print("Loaded existing Ensemble Voting Classifier.")
        except Exception as e:
            print(f"Error loading ensemble model: {e}. Skipping ensemble evaluation.")
            trained_models['EnsembleVotingClassifier'] = None


    print("\n[Step 3/7] Generating Visualizations...")
    # Combine PCA data for plotting
    X_combined_pca = np.vstack((X_train_pca, X_test_pca))
    y_combined = np.hstack((y_train, y_test))

    # 4. Visualize Decision Boundaries for all models (using PCA)
    for model_name, model in trained_models.items():
        if model is not None:
            # Skip ensemble for decision boundary plots directly, as it's a meta-model
            if model_name != 'EnsembleVotingClassifier':
                plot_decision_boundary(X_combined_pca, y_combined, model,
                                    f'{model_name} Decision Boundary on Heart Disease Data (PCA)',
                                    f'{model_name.lower().replace(" ", "_")}_decision_boundary_heart_v1.png',
                                    plot_feature_names, PLOTS_PATH)

    print("\n[Step 5/7] Generating Confusion Matrices...")
    # DEBUG: Print shape of X_test and model's expected features just before prediction
    print(f"DEBUG: Shape of X_test used for Confusion Matrix generation: {X_test.shape}")
    # 5. Generate and Plot Confusion Matrices for all models
    for model_name, model in trained_models.items():
        if model is not None:
            if hasattr(model, 'n_features_in_'):
                print(f"DEBUG: {model_name} model (type: {type(model).__name__}) expects {model.n_features_in_} features.")
            else:
                print(f"DEBUG: {model_name} model (type: {type(model).__name__}) does not have n_features_in_ attribute. This is common for ensemble models or older sklearn versions.")

            y_pred = model.predict(X_test)
            plot_confusion_matrix(y_test, y_pred, model_name,
                                  f'confusion_matrix_{model_name.lower().replace(" ", "_")}_v1.png',
                                  PLOTS_PATH)

    print("\n[Step 6/7] Generating ROC AUC Curves...")
    # 6. Generate and Plot ROC AUC Curves for all models
    for model_name, model in trained_models.items():
        if model is not None:
            try:
                # Use predict_proba for ROC if available (should be for all models now)
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test)[:, 1]
                else:
                    # Fallback for models that might not have predict_proba (e.g., some custom estimators)
                    y_scores = model.decision_function(X_test) # If decision_function exists
                
                plot_roc_curve(y_test, y_scores, model_name,
                               f'roc_curve_{model_name.lower().replace(" ", "_")}_v1.png',
                               PLOTS_PATH)
            except Exception as e:
                print(f"Could not generate ROC curve for {model_name}: {e}")
                print("Ensure the model has 'predict_proba' method or a compatible 'decision_function'.")


    print("\n--- Machine Learning Classification Pipeline Completed Successfully! ---")
    print("Check 'models/', 'results/', and 'results/plots/' directories for outputs.")

if __name__ == "__main__":
    main()
