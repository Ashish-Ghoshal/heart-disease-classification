"""
visualizer.py

This script contains functions for visualizing SVM decision boundaries,
confusion matrices, and ROC AUC curves.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.svm import SVC # Only needed for type hinting/temporary model creation for plots
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_decision_boundary(X_pca, y, model, title, filename, feature_names, plots_path):
    """
    Plots the decision boundary of an SVM classifier using PCA components.
    A temporary 2D model is trained for visualization purposes, matching the
    best hyperparameters of the full model.

    Args:
        X_pca (np.array): PCA-transformed feature data (2D).
        y (np.array): Target labels.
        model (sklearn.svm.SVC): The best trained SVM classifier (trained on full features).
        title (str): Title for the plot.
        filename (str): Name to save the plot file.
        feature_names (list): Names of the two PCA components for axis labels.
        plots_path (str): Directory path to save plot files.
    """
    print(f"\n[Step 3/X] Visualizing Decision Boundary for {title.split(' ')[0]} SVM...")

    # Create a mesh to plot in based on the 2D PCA data
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # To visualize the decision boundary of a model trained on high-dimensional data in 2D PCA space,
    # we retrain a temporary 2D model on the PCA-transformed data that has the same best hyperparameters.
    print(f"Retraining a temporary 2D {model.kernel} SVM for visualization based on PCA components.")
    # Use the best C and gamma from the full model
    # Note: 'gamma' parameter in SVC is 'scale' by default for RBF if not specified,
    # but for explicit control and matching, we use it if kernel is RBF.
    temp_model_for_plot = SVC(kernel=model.kernel, C=model.C,
                              gamma=model.gamma if model.kernel == 'rbf' else 'scale',
                              random_state=42, probability=True) # probability=True for consistency if needed
    temp_model_for_plot.fit(X_pca, y) # Fit temporary model on 2D PCA data

    Z = temp_model_for_plot.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    # Plot the data points (which are already PCA-transformed)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=80)
    plt.xlabel(f'{feature_names[0]}')
    plt.ylabel(f'{feature_names[1]}')
    plt.title(title)
    plt.colorbar(scatter, ticks=[0, 1], label='Heart Disease (0: No, 1: Yes)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    plot_save_path = os.path.join(plots_path, filename)
    plt.savefig(plot_save_path)
    plt.close() # Close the plot to free memory
    print(f"Decision boundary plot saved to: {plot_save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name, filename, plots_path):
    """
    Plots and saves the confusion matrix.

    Args:
        y_true (np.array): True target labels.
        y_pred (np.array): Predicted labels from the model.
        model_name (str): Name of the model for the plot title.
        filename (str): Name to save the plot file.
        plots_path (str): Directory path to save plot files.
    """
    print(f"\n[Step 4/X] Plotting Confusion Matrix for {model_name}...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Disease', 'Predicted Disease'],
                yticklabels=['Actual No Disease', 'Actual Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plot_save_path = os.path.join(plots_path, filename)
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Confusion Matrix plot saved to: {plot_save_path}")


def plot_roc_curve(y_true, y_scores, model_name, filename, plots_path):
    """
    Plots and saves the ROC curve.

    Args:
        y_true (np.array): True target labels.
        y_scores (np.array): Target scores, probabilities of the positive class.
        model_name (str): Name of the model for the plot title.
        filename (str): Name to save the plot file.
        plots_path (str): Directory path to save plot files.
    """
    print(f"\n[Step 5/X] Plotting ROC Curve for {model_name}...")
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_save_path = os.path.join(plots_path, filename)
    plt.savefig(plot_save_path)
    plt.close()
    print(f"ROC Curve plot saved to: {plot_save_path}")
