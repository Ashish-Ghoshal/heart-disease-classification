"""
data_loader.py

This script handles loading, preprocessing, and splitting the Heart Disease UCI dataset.
It applies imputation, scaling, one-hot encoding, and PCA for visualization.
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

def load_and_preprocess_data(data_path, dataset_file='heart.csv', random_state=42):
    """
    Loads the Heart Disease UCI dataset, preprocesses it for binary classification,
    splits it into training and testing sets, and scales features.
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
        ])

    # Fit and transform the full dataset using the preprocessor
    X_processed = preprocessor.fit_transform(X_full)

    print(f"\nShape of fully preprocessed X for training: {X_processed.shape}")
    print(f"Shape of y: {y.shape}")

    # Split the fully preprocessed data for model training
    X_train_processed, X_test_processed, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=random_state, stratify=y
    )

    print(f"\nTraining set shape: X_train={X_train_processed.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test_processed.shape}, y_test={y_test.shape}")

    # Prepare data for 2D visualization using PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_processed) # Fit PCA on the entire processed dataset

    # Split the PCA-transformed data
    X_train_pca, X_test_pca, _, _ = train_test_split(
        X_pca, y, test_size=0.3, random_state=random_state, stratify=y
    )
    plot_feature_names = ['Principal Component 1', 'Principal Component 2']

    print(f"Shape of PCA-transformed X for visualization: {X_pca.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, X_train_pca, X_test_pca, plot_feature_names
