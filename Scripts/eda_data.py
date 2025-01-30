import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_information(df: pd.DataFrame):
    print("Data Shape:")
    print(df.shape)
    
    print("Data Describe:")
    print(df.describe())  
    
    print("Data Info:")
    print(df.info())
def feature_connection(df):
    numerical_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10,8))
    sns.heatmap(numerical_df.corr(), cmap="RdBu")
    plt.title("Correlations Between Variables", size=15)
    plt.show()

def distributed_feature(feature):
    if not feature.dtype in ['int64', 'float64']:
        print("Error: The provided feature is not numeric.")
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(feature, kde=True)
    plt.title("Feature Distribution")
    plt.show()
    # Compute and print skewness and kurtosis
    print("Skewness: %f" % feature.skew())
    print("Kurtosis: %f" % feature.kurt())
