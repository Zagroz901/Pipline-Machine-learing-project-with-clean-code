from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
from evaluate import * 

def lin_reg(x_train,y_train,x_test,y_test,models):
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    predictions = lin_reg.predict(x_test)

    mae, mse, rmse, r_squared = evaluation(y_test, predictions)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r_squared)
    print("-"*30)
    # rmse_cross_val = rmse_cv(lin_reg)
    print("RMSE Cross-Validation:", rmse_cross_val)
    new_row = {"Model": "LinearRegression","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
    models = models.append(new_row, ignore_index=True)