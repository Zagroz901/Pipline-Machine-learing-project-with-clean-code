from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

def dataTarget(df):
    data = df.drop(["OutcomeVariable"], axis=1)
    target = df["OutcomeVariable"]
    return data,target

def split_to_category(df):
    numeric_col=df.select_dtypes(include=['int64','float64']).columns.tolist()
    categoric_col=df.select_dtypes(include=['object']).columns.tolist()
    return numeric_col,categoric_col

def fill_null_train(df,numeric_col,categoric_col):
    imputer = SimpleImputer(strategy='median')
    df[numeric_col] = imputer.fit_transform(df[numeric_col])
    imputer = SimpleImputer(strategy='most_frequent')
    df[categoric_col] = imputer.fit_transform(df[categoric_col])

def one_hot(df,categoric_col):
    X = pd.get_dummies(df, columns=categoric_col)
    return X

def standard_scallar(df,numeric_col):
    scaler = StandardScaler()
    df[numeric_col] = scaler.fit_transform(df[numeric_col])

