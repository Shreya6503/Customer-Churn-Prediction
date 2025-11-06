import pandas as pd

def preprocess_data(df):
    df.columns = df.columns.str.strip().str.replace(' ', '').str.lower()

    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df['totalcharges'].fillna(df['totalcharges'].median(), inplace=True)

    df['churn'] = df['churn'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    if 'customerid' in df.columns:
        df.drop(['customerid'], axis=1, inplace=True)

    return df
