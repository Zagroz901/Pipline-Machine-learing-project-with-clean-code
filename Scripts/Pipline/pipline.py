import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from Model.model import *
from Procces.feature_selection import * 
from Procces.clean_data import *
# === Load Data ===
def load_data(filepath):
    return pd.read_csv(filepath)

# === Preprocessing Pipeline ===
def build_preprocessing_pipeline(numeric_features, categorical_features):
    """Creates a pipeline for missing value handling, scaling, and encoding."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # Fill missing numeric values
        ("scaler", StandardScaler())  # Standardize
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing categorical values
        ("encoder", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    
    return preprocessor


def run_pipeline(data_path):
    # Load data
    df = load_data(data_path)
    missing_data = NullInEachFeature(df)
    df_clean = DropTotal(df,missing_data)

    # Define target and features
    target_col = "OutcomeVariable"  # Update to your actual target column
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # Identify numerical and categorical features
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

    # Transform data
    X_transformed = preprocessor.fit_transform(X)

    # ✅ Convert sparse matrix to dense NumPy array
    X_transformed = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Initialize DataFrame for results
    models_df = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"])
    trained_models = {}

    # List of models
    models_list = [
        LinearRegression(),
        Ridge(alpha=1.0),
        RandomForestRegressor(n_estimators=100),
        ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        Lasso(alpha=0.1, max_iter=5000),
        SVR(kernel="rbf", C=1.0, epsilon=0.1),
        XGBRegressor(n_estimators=100, learning_rate=0.1)
    ]

    # Train and evaluate models
    for model in models_list:
        models_df, trained_models = train_and_evaluate(model, X_train, y_train, X_test, y_test, models_df, trained_models)

    return models_df, trained_models



