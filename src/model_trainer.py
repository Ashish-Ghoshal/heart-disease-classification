

import os
import joblib
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def train_and_tune_svm(X_train, y_train, X_test, y_test, kernel_type, models_path, results_path):
    """
        Train and tune an SVM ('linear' or 'rbf'), save the best model and results.

        Args:
            X_train, y_train: Training data.
            X_test, y_test: Test data.
            kernel_type: 'linear' or 'rbf'.
            models_path: Path to save model.
            results_path: Path to save results.

        Returns:
            Best SVM model, classification report dict.
    """
    print(f"\n[Step 2/X] Training and Tuning {kernel_type.upper()} SVM...")

    if kernel_type == 'linear':
        # Expanded param_grid for Linear SVM
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
        svm_model = SVC(kernel='linear', random_state=42, probability=True) # probability=True for ROC curves
    elif kernel_type == 'rbf':
        # Expanded param_grid for RBF SVM
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000, 10000],
            'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }
        svm_model = SVC(kernel='rbf', random_state=42, probability=True) # probability=True for ROC curves
    else:
        raise ValueError("Invalid kernel_type. Must be 'linear' or 'rbf'.")

    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        cv=5,                 # Set cross-validation folds back to 5
        scoring='accuracy',   # Metric to optimize
        n_jobs=-1,            # Use all available CPU cores
        verbose=2             # Increased verbose output
    )

    grid_search.fit(X_train, y_train)
    best_svm = grid_search.best_estimator_

    print(f"\nBest parameters for {kernel_type.upper()} SVM: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {kernel_type.upper()} SVM: {grid_search.best_score_:.4f}")

    y_pred = best_svm.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy for {kernel_type.upper()} SVM: {accuracy:.4f}")
    print(f"Classification Report for {kernel_type.upper()} SVM:\n")
    print(classification_report(y_test, y_pred))

    # Save the best model
    model_filename = os.path.join(models_path, f'{kernel_type}_svm_model_v1.joblib')
    joblib.dump(best_svm, model_filename)
    print(f"Best {kernel_type.upper()} SVM model saved to: {model_filename}")

    # Prepare cv_results_ for JSON serialization

    serializable_cv_results = {k: v.tolist() if isinstance(v, np.ndarray) else v
                               for k, v in grid_search.cv_results_.items()}

    # Save tuning results and classification report as JSON
    tuning_results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'cv_results': serializable_cv_results
    }
    with open(os.path.join(results_path, f'tuning_results_{kernel_type}_v1.json'), 'w') as f:
        json.dump(tuning_results, f, indent=4)

    with open(os.path.join(results_path, f'classification_report_{kernel_type}_v1.json'), 'w') as f:
        json.dump(report, f, indent=4)

    print(f"{kernel_type.upper()} SVM tuning results and classification report saved.")

    return best_svm, report
