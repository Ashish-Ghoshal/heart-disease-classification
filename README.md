# Heart Disease Classification

# Table of Contents

-   [Project Overview](#project-overview)
-   [Key Features](#key-features)
-   [Technologies and Libraries Used](#technologies-and-libraries-used)
-   [Project Structure](#project-structure)
-   [Setup and Execution (Local)](#setup-and-execution-local)
    -   [1. Prerequisites](#1-prerequisites)
    -   [2. Clone the Repository](#2-clone-the-repository)
    -   [3. Create and Activate Conda Environment](#3-create-and-activate-conda-environment)
    -   [4. Install Dependencies](#4-install-dependencies)
    -   [5. Data Preparation](#5-data-preparation)
    -   [6. Run the Project](#6-run-the-project)
-   [How to Use and Interpret Results](#how-to-use-and-interpret-results)
-   [Future Enhancements](#future-enhancements)
-   [Contributing](#contributing)
-   [License](#license)

## Project Overview

This repository hosts a machine learning project focused on demonstrating and implementing Support Vector Machines (SVMs) for binary classification of heart disease. The project addresses the fundamental concepts of SVMs, including linear and non-linear classification using different kernels (Linear and Radial Basis Function - RBF), hyperparameter tuning, and performance evaluation through cross-validation.

The primary goal is to provide a clear, reproducible, and well-documented example of applying SVMs to a dataset, visualizing their decision boundaries in a 2D space (using PCA), and optimizing their performance. This project serves as a foundational understanding of SVMs for machine learning enthusiasts and practitioners.

## Key Features

*   **Data Loading & Preprocessing:** Handles loading the Heart Disease UCI dataset, including imputation for missing values, scaling numerical features, and one-hot encoding categorical features.
    
*   **Linear SVM:** Implementation and training of a Support Vector Classifier with a linear kernel.
    
*   **RBF Kernel SVM:** Implementation and training of a Support Vector Classifier with an RBF (Gaussian) kernel for non-linear decision boundaries.
    
*   **Decision Boundary Visualization:** Generates plots to visualize the decision boundaries learned by both linear and RBF SVMs on 2D PCA-transformed data.
    
*   **Hyperparameter Tuning:** Utilizes `GridSearchCV` for systematic and extended hyperparameter tuning (e.g., `C` and `gamma`) to optimize model performance.
    
*   **Cross-Validation:** Employs 5-fold cross-validation for robust model evaluation and to prevent overfitting.
    
*   **Performance Metrics:** Calculates and reports standard classification metrics such as accuracy, precision, recall, and F1-score.
    
*   **Comprehensive Evaluation Plots:** Generates and saves Confusion Matrices and Receiver Operating Characteristic (ROC) curves with Area Under the Curve (AUC).
    
*   **Structured Output:** Saves trained models and evaluation results (including best hyperparameters and full cross-validation results) in a structured format (JSON files) and plots as image files.
    

## Technologies and Libraries Used

*   **Python 3.x:** The core programming language.
    
*   **Scikit-learn:** For SVM implementations, preprocessing, dimensionality reduction (PCA), model selection, and evaluation metrics.
    
*   **NumPy:** For numerical operations and array manipulation.
    
*   **Matplotlib:** For creating static plots, especially for decision boundaries, confusion matrices, and ROC curves.
    
*   **Seaborn:** For enhanced data visualizations, particularly for confusion matrices.
    
*   **Pandas:** For data manipulation and analysis.
    
*   **Joblib:** For saving and loading trained models.
    
*   **JSON:** For saving results and metadata in a structured format.
    

## Project Structure

    .
    ├── data/
    │   └── heart.csv                  # The Heart Disease UCI dataset
    ├── models/
    │   └── linear_svm_model_v1.joblib    # Example saved linear SVM model
    │   └── rbf_svm_model_v1.joblib       # Example saved RBF SVM model
    ├── results/
    │   ├── tuning_results_linear_v1.json # Hyperparameter tuning results for linear SVM
    │   ├── tuning_results_rbf_v1.json    # Hyperparameter tuning results for RBF SVM
    │   ├── classification_report_linear_v1.json # Classification report for linear SVM
    │   ├── classification_report_rbf_v1.json    # Classification report for RBF SVM
    │   └── plots/
    │       ├── linear_decision_boundary_heart_v1.png  # PCA visualization of linear SVM decision boundary
    │       ├── rbf_decision_boundary_heart_v1.png     # PCA visualization of RBF SVM decision boundary
    │       ├── confusion_matrix_linear_v1.png         # Confusion matrix for linear SVM
    │       ├── confusion_matrix_rbf_v1.png            # Confusion matrix for RBF SVM
    │       ├── roc_curve_linear_v1.png                # ROC curve for linear SVM
    │       └── roc_curve_rbf_v1.png                   # ROC curve for RBF SVM
    ├── src/
    │   ├── data_loader.py              # Script for loading, preprocessing, and splitting data
    │   ├── model_trainer.py            # Script for training SVM models and hyperparameter tuning
    │   └── visualizer.py               # Script for plotting decision boundaries and evaluation metrics
    ├── main.py                       # Main script to run the entire workflow locally
    ├── requirements.txt              # List of Python dependencies
    └── .gitignore                    # Specifies intentionally untracked files to ignore
    

## Setup and Execution (Local)

Follow these steps to set up and run the project on your local machine.

### 1\. Prerequisites

*   **Conda:** Ensure you have Anaconda or Miniconda installed. If not, download it from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html "null").
    

### 2\. Clone the Repository

    git clone https://github.com/yourusername/heart-disease-classification.git
    cd heart-disease-classification
    

### 3\. Create and Activate Conda Environment

It's highly recommended to use a virtual environment to manage dependencies.

    conda create -n svm_env python=3.9
    conda activate svm_env
    

### 4\. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

    pip install -r requirements.txt
    

### 5\. Data Preparation

*   Download the `heart.csv` file from the [Heart Disease UCI dataset on Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data "null").
    
*   Place the downloaded `heart.csv` file inside the `data/` directory of your cloned repository.
    

### 6\. Run the Project

Execute the `main.py` script to run the entire machine learning pipeline.

    python main.py
    

This script will:

*   Load and preprocess the data, including imputation, scaling, one-hot encoding, and PCA for visualization.
    
*   Train linear and RBF SVM models with extended hyperparameter tuning.
    
*   Evaluate model performance using various metrics.
    
*   Save trained models, detailed tuning results (JSON), classification reports (JSON), and all generated plots (PNG) in their respective directories.
    

## How to Use and Interpret Results

After running `main.py`, the `models/` and `results/` directories will be populated:

*   **`models/`**: Contains the trained SVM models (`.joblib` files). You can load these models for future predictions without retraining.
    
*   **`results/tuning_results_*.json`**: These files provide detailed insights into the hyperparameter tuning process, including the best parameters found and cross-validation scores for each parameter combination.
    
*   **`results/classification_report_*.json`**: These files offer a comprehensive breakdown of model performance on the test set, including precision, recall, F1-score, and support for each class (No Disease/Disease).
    
*   **`results/plots/`**: This directory contains PNG images visualizing various aspects of the models:
    
    *   **Decision Boundary Plots:** Show how the SVM models separate the data points in the 2-dimensional PCA space. For the RBF kernel, you'll likely see a non-linear boundary, demonstrating its ability to handle complex separations.
        
    *   **Confusion Matrices:** Illustrate the number of correct and incorrect predictions made by the models, broken down by class.
        
    *   **ROC AUC Curves:** Provide a visual representation of the models' ability to distinguish between classes across various classification thresholds. A higher Area Under the Curve (AUC) indicates better overall performance.
        

By examining these outputs, you can gain a deep understanding of your SVM models' strengths and weaknesses, and compare the performance of the linear and RBF kernels on the Heart Disease UCI dataset.

## Future Enhancements

To make this project more robust, scalable, and valuable in a real-world context, consider the following enhancements:

*   **Automated Data Download:** Integrate functionality to automatically download datasets from common sources (e.g., Kaggle API) instead of manual placement.
    
*   **Experiment Tracking:** Implement an experiment tracking system (e.g., MLflow, ClearML, Weights & Biases) to log parameters, metrics, models, and artifacts for better experiment management and reproducibility. This is crucial for comparing multiple model runs and versions.
    
*   **More Advanced Hyperparameter Optimization:** Explore advanced hyperparameter optimization libraries like `Optuna` or `Hyperopt` for more efficient and sophisticated tuning strategies, especially for larger search spaces or more complex models.
    
*   **Ensemble Methods/Meta-Learning:** Investigate combining multiple SVMs or integrating SVMs into ensemble methods (e.g., stacking, boosting) to potentially achieve higher predictive accuracy.
    
*   **Feature Engineering Module:** Develop a dedicated module for feature engineering, allowing for the creation of new features from existing ones, which can significantly improve model performance, especially for non-linear problems.
    
*   **Deployment Readiness:** Containerize the application using Docker, making it easy to deploy to various environments (e.g., cloud platforms, local servers) and ensuring consistent behavior across different setups.
    
*   **REST API for Predictions:** Build a simple Flask/FastAPI application around the trained model to expose predictions via a REST API, enabling integration with other applications or services.
    
*   **Interactive Visualizations:** Upgrade static Matplotlib plots to interactive visualizations using libraries like Plotly or Bokeh, allowing users to explore decision boundaries and data distributions dynamically.
    
*   **Uncertainty Quantification:** Implement methods to quantify the uncertainty in predictions, which is critical for high-stakes applications where knowing the confidence of a prediction is as important as the prediction itself.
    

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to:

1.  Fork the repository.
    
2.  Create a new branch (`git checkout -b feature/your-feature-name` or `bugfix/issue-description`).
    
3.  Make your changes and ensure tests (if any) pass.
    
4.  Commit your changes (`git commit -m 'Add new feature'`).
    
5.  Push to the branch (`git push origin feature/your-feature-name`).
    
6.  Open a Pull Request.
    

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. (Note: A `LICENSE` file would be included in a real repo, typically with the MIT license boilerplate).