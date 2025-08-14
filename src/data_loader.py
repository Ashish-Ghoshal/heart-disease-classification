"""
data_loader.py

This script handles loading, preprocessing, and splitting the Heart Disease UCI dataset.
It applies imputation, scaling, one-hot encoding, feature selection, and PCA for visualization.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier # ADDITION v12: For feature selection
from sklearn.feature_selection import SelectFromModel # ADDITION v12: For feature selection

def load_and_preprocess_data(data_path, dataset_file='heart.csv', random_state=42):
    """
    Loads the Heart Disease UCI dataset, preprocesses it for binary classification,
    applies feature selection, splits it into training and testing sets, and scales features.
    It also prepares a PCA-transformed version of the data for 2D visualization.

    Args:
        data_path (str): The directory path where the dataset file is located.
        dataset_file (str): Name of the dataset CSV file.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train_processed, X_test_processed, y_train, y_test,
                X_train_pca, X_test_pca, plot_feature_names)
    """
    file_path = os.path.join(data_path, dataset_file)
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset '{dataset_file}' loaded.")
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        print("Please ensure 'heart.csv' is placed in the 'data/' directory of your repository.")
        exit() # Exit if dataset not found

    # Convert 'num' to binary target: 0 (no disease) and 1 (disease present)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    # Drop the original 'num' column, 'id', and 'dataset' as they are not features for modeling
    df_cleaned = df.drop(columns=['num', 'id', 'dataset'])

    # Identify categorical and numerical features
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_features = [
        'age', 'trestbps', 'chol', 'thalch', 'oldpeak'
    ]

    # Separate features (X_full) and target (y)
    X_full = df_cleaned.drop(columns=['target'])
    y = df_cleaned['target'].values

    print(f"\nOriginal number of features: {X_full.shape[1]}")
    print(f"Target variable 'target' value counts:\n{pd.Series(y).value_counts()}")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though not expected here
    )

    # Fit and transform the full dataset using the preprocessor
    X_processed_initial = preprocessor.fit_transform(X_full)

    print(f"\nShape of initially preprocessed X: {X_processed_initial.shape}")

    # ADDITION v12: Feature Selection using RandomForest Importance
    # We will use a RandomForestClassifier to get feature importances
    # and then select features based on a threshold.
    print("Performing Feature Selection...")
    feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=random_state),
                                       threshold='median') # Select features with importance > median
    feature_selector.fit(X_processed_initial, y)
    X_processed = feature_selector.transform(X_processed_initial)
    print(f"Shape of X after feature selection: {X_processed.shape}")
    print(f"Number of features selected: {X_processed.shape[1]}")

    print(f"Shape of fully preprocessed X for training: {X_processed.shape}")
    print(f"Shape of y: {y.shape}")

    # Split the fully preprocessed data for model training
    X_train_processed, X_test_processed, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=random_state, stratify=y
    )

    print(f"\nTraining set shape: X_train={X_train_processed.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test_processed.shape}, y_test={y_test.shape}")

    # Prepare data for 2D visualization using PCA on the selected features
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_processed) # Apply PCA on the feature-selected dataset

    # Split the PCA-transformed data
    X_train_pca, X_test_pca, _, _ = train_test_split(
        X_pca, y, test_size=0.3, random_state=random_state, stratify=y
    )
    plot_feature_names = ['Principal Component 1', 'Principal Component 2']

    print(f"Shape of PCA-transformed X for visualization: {X_pca.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, X_train_pca, X_test_pca, plot_feature_names
