from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
# Function to calculate RMSE using cross-validation
def rmse_cv(model, X, y, cv=5):
    """Computes Root Mean Squared Error using cross-validation, handling XGBoost compatibility."""
    X = np.array(X)  # Ensure X is a NumPy array
    y = np.array(y)  # Ensure y is a NumPy array
    
    try:
        rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)).mean()
    except Exception as e:
        print(f"Error in cross-validation for {type(model).__name__}: {e}")
        rmse = None  # Handle errors gracefully
    return rmse


# General evaluation function
def evaluation(y, predictions):
    """Evaluates regression model performance using MAE, MSE, RMSE, and R-squared."""
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared

# Generalized function to train & evaluate any regression model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, models_df,trained_models, model_name=None):
    """
    Trains a regression model, evaluates its performance, and appends results to models_df.

    Parameters:
    - model: Any regression model instance (e.g., LinearRegression(), Ridge(), etc.)
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - models_df: DataFrame to store model evaluation results
    - model_name: Optional, specify a custom name for the model in the results table

    Returns:
    - Updated models DataFrame
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate performance
    mae, mse, rmse, r_squared = evaluation(y_test, predictions)
    
    # Compute RMSE using cross-validation
    rmse_cross_val = rmse_cv(model, X_train, y_train)

    # Determine model name
    if model_name is None:
        model_name = type(model).__name__  # Get class name dynamically
        trained_models[model_name]=model

    # # Print results
    # print(f"Model: {model_name}")
    # print("MAE:", mae)
    # print("MSE:", mse)
    # print("RMSE:", rmse)
    # print("R2 Score:", r_squared)
    # print("RMSE Cross-Validation:", rmse_cross_val)
    # print("-" * 40)
    # Append results to DataFrame
    new_row = {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r_squared,
        "RMSE (Cross-Validation)": rmse_cross_val
    }
    models_df = pd.concat([models_df, pd.DataFrame([new_row])], ignore_index=True)

    return models_df,trained_models
