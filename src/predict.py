import joblib
import pandas as pd

model = joblib.load('models/model_xgb.pkl')

def predict_churn(data: dict):

    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob > 0.35)

    return {
        'probability': float(prob),
        'prediction': int(pred)
    }



if __name__ == "__main__":
    sample = {
        "gender": "Female",
        "partner": "Yes",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "senior_citizen": 0,
        "tenure": 5,
        "monthly_charges": 75.4,
        "total_charges": 350.2
    }

    print(predict_churn(sample))

