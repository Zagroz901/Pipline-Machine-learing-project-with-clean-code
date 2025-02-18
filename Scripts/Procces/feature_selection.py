import pandas as pd

def NullInEachFeature(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/ df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
