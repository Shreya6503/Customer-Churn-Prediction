import joblib
import pandas as pd

model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_features = joblib.load("models/model_features.pkl")

def predict_churn(customer_data):
    df = pd.DataFrame([customer_data])

    df = pd.get_dummies(df, drop_first=True)

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]

    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    return pred, prob
