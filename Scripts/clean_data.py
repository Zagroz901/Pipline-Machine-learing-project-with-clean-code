def DropTotal(df,missing_data):
    df_clean = df.drop((missing_data[missing_data['Total'] >= 500]).index, axis=1)
    return df_clean
