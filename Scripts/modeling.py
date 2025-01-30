from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
# from xgboost import XGBRegressor
import joblib
import numpy as np

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared

def lin_reg(x_train,y_train,x_test,y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    predictions = lin_reg.predict(x_test)

    mae, mse, rmse, r_squared = evaluation(y_test, predictions)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r_squared)
    print("-"*30)
    rmse_cross_val = rmse_cv(lin_reg)
    print("RMSE Cross-Validation:", rmse_cross_val)
    new_row = {"Model": "LinearRegression","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
    models = models.append(new_row, ignore_index=True)
def RidgeModel(x_train,y_train,x_test,y_test):
    ridge = Ridge()
    ridge.fit(x_train, y_train)
    predictions = ridge.predict(x_test)
    mae, mse, rmse, r_squared = evaluation(y_test, predictions)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r_squared)
    print("-"*30)
    rmse_cross_val = rmse_cv(ridge)
    print("RMSE Cross-Validation:", rmse_cross_val)

    new_row = {"Model": "Ridge","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
    models = models.append(new_row, ignore_index=True)