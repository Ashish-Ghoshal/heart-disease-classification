

import os
import sys
import numpy as np

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from data_loader import load_and_preprocess_data
from model_trainer import train_and_tune_svm
from visualizer import plot_decision_boundary, plot_confusion_matrix, plot_roc_curve

def main():
    """
    Main function to run the SVM classification pipeline.
    """
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    MODELS_PATH = os.path.join(BASE_DIR, 'models')
    RESULTS_PATH = os.path.join(BASE_DIR, 'results')
    PLOTS_PATH = os.path.join(RESULTS_PATH, 'plots')

    # Ensure directories exist
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    print("--- Starting SVM Classification Pipeline ---")
    print(f"Data will be loaded from: {DATA_PATH}")
    print(f"Models will be saved to: {MODELS_PATH}")
    print(f"Results (JSON, plots) will be saved to: {RESULTS_PATH} and {PLOTS_PATH}")

    # 1. Load and preprocess data
    print("\n[Step 1/7] Loading and Preprocessing Data...")
    (X_train, X_test, y_train, y_test,
     X_train_pca, X_test_pca, plot_feature_names) = load_and_preprocess_data(DATA_PATH)

    # Define model paths for checking existence
    linear_model_path = os.path.join(MODELS_PATH, 'linear_svm_model_v1.joblib')
    rbf_model_path = os.path.join(MODELS_PATH, 'rbf_svm_model_v1.joblib')

    best_linear_svm = None
    best_rbf_svm = None

    
    retrain_models = 'y' # Default to 'y' for initial run or if no models found
    
    if os.path.exists(linear_model_path) and os.path.exists(rbf_model_path):
        print("\nExisting trained models found.")
        user_input = input("Do you want to retrain the models? (y/n): ").lower()
        if user_input == 'n':
            retrain_models = 'n'
            try:
                import joblib
                best_linear_svm = joblib.load(linear_model_path)
                best_rbf_svm = joblib.load(rbf_model_path)
                print("Using existing trained models.")
            except Exception as e:
                print(f"Error loading existing models: {e}. Retraining models.")
                retrain_models = 'y'
        elif user_input != 'y':
            print("Invalid input. Defaulting to retraining models.")

    if retrain_models == 'y':
        # 2. Train and tune Linear SVM
        best_linear_svm, linear_report = train_and_tune_svm(
            X_train, y_train, X_test, y_test, 'linear', MODELS_PATH, RESULTS_PATH
        )

        # 3. Train and tune RBF SVM
        best_rbf_svm, rbf_report = train_and_tune_svm(
            X_train, y_train, X_test, y_test, 'rbf', MODELS_PATH, RESULTS_PATH
        )
    else:

        print("Skipping model retraining. Proceeding with visualization using loaded models.")


    # Ensure models are available for subsequent steps
    if best_linear_svm is None or best_rbf_svm is None:
        print("Models are not available for visualization and evaluation. Please ensure training or loading was successful.")
        sys.exit(1) # Exit if models are not ready


    # Combine PCA data for plotting
    X_combined_pca = np.vstack((X_train_pca, X_test_pca))
    y_combined = np.hstack((y_train, y_test))


    print("\n[Step 4/7] Generating Visualizations...")
    # 4. Visualize Linear SVM decision boundary
    plot_decision_boundary(X_combined_pca, y_combined, best_linear_svm,
                           'Linear SVM Decision Boundary on Heart Disease Data (PCA)',
                           'linear_decision_boundary_heart_v1.png', plot_feature_names, PLOTS_PATH)

    # 5. Visualize RBF SVM decision boundary
    plot_decision_boundary(X_combined_pca, y_combined, best_rbf_svm,
                           'RBF SVM Decision Boundary on Heart Disease Data (PCA)',
                           'rbf_decision_boundary_heart_v1.png', plot_feature_names, PLOTS_PATH)

    print("\n[Step 6/7] Generating Confusion Matrices...")
    # 6. Generate and Plot Confusion Matrices
    y_pred_linear = best_linear_svm.predict(X_test)
    y_pred_rbf = best_rbf_svm.predict(X_test)

    plot_confusion_matrix(y_test, y_pred_linear, 'Linear SVM', 'confusion_matrix_linear_v1.png', PLOTS_PATH)
    plot_confusion_matrix(y_test, y_pred_rbf, 'RBF SVM', 'confusion_matrix_rbf_v1.png', PLOTS_PATH)

    print("\n[Step 7/7] Generating ROC AUC Curves...")
    # 7. Generate and Plot ROC AUC Curves
    try:
        y_scores_linear = best_linear_svm.predict_proba(X_test)[:, 1]
        y_scores_rbf = best_rbf_svm.predict_proba(X_test)[:, 1]

        plot_roc_curve(y_test, y_scores_linear, 'Linear SVM', 'roc_curve_linear_v1.png', PLOTS_PATH)
        plot_roc_curve(y_test, y_scores_rbf, 'RBF SVM', 'roc_curve_rbf_v1.png', PLOTS_PATH)
    except Exception as e:
        print(f"Could not generate ROC curves for local execution: {e}")
        print("Ensure 'probability=True' was set when initializing SVC models during training.")


    print("\n--- SVM Classification Pipeline Completed Successfully! ---")
    print("Check 'models/', 'results/', and 'results/plots/' directories for outputs.")

if __name__ == "__main__":
    main()
