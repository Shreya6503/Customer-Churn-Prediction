import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load("../models/churn_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
model_features = joblib.load("../models/model_features.pkl")

def predict_churn(customer_data: pd.DataFrame):
    # Encode categorical features
    customer_data = pd.get_dummies(customer_data, drop_first=True)

    # Add missing columns
    missing = set(model_features) - set(customer_data.columns)
    for col in missing:
        customer_data[col] = 0

    customer_data = customer_data[model_features]

    # Scale input
    scaled_data = scaler.transform(customer_data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[:, 1][0]

    return prediction, probability

# Usage Example:
# from utils import predict_churn
# import pandas as pd

# customer = pd.DataFrame({
#     "tenure": [4],
#     "MonthlyCharges": [70],
# })
# pred, prob = predict_churn(customer)
# print(pred, prob)
