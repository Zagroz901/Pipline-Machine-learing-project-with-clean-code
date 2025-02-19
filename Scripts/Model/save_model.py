import joblib
import pandas as pd

# === Function to Select & Save the Best Model ===
def save_best_model(models_df, trained_models, save_path="best_model.pkl"):
    """
    Selects the best model based on the lowest RMSE and saves it as a .pkl file.

    Parameters:
    - models_df: DataFrame containing model performance metrics.
    - trained_models: Dictionary of trained models (keys = model names, values = model objects).
    - save_path: Path to save the best model.

    Returns:
    - The name of the best model saved.
    """
    # Select the best model based on the lowest RMSE
    best_model_row = models_df.loc[models_df["RMSE"].idxmin()]  # Get row with lowest RMSE
    best_model_name = best_model_row["Model"]
    
    # Get the corresponding trained model object
    best_model = trained_models[best_model_name]

    # Save the best model as a .pkl file
    joblib.dump(best_model, save_path)
    print(f"✅ Best model ({best_model_name}) saved as {save_path} with RMSE: {best_model_row['RMSE']}")

    return best_model_name

