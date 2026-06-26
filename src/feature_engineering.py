def create_features(df):
    df['Debt_to_Income_Ratio'] = df['Debt']/df['Income']
    return df