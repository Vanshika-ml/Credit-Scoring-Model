import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    df.fillna(df.mean(numeric_only=True), inplace=True)

    le = LabelEncoder()

    df["Payment_History"] = le.fit_transform(df["Payment_History"])

    return df