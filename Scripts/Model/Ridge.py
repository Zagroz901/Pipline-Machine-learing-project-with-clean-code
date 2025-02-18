from sklearn.linear_model import Ridge
import numpy as np
from evaluate import * 

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