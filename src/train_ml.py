import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import os


# Load Dataset
# -----------------------

df = pd.read_csv('data/processed/final_ml_dataset.csv')


# Feature Columns
# -----------------------

categorical_cols = [
    "gender", "partner", "dependents",
    "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method"
]

numeric_cols = [
    "senior_citizen", "tenure",
    "monthly_charges", "total_charges"
]


X = df[categorical_cols + numeric_cols]
y = df['churn']


# Train-Test split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42, stratify = y)

# Class Imbalance Handling
# -----------------------

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train ==1])

# Preprocessing pipeline
# -----------------------

preprocessor = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# Final XGBoost Model Pipeline
# -----------------------

model = Pipeline([
    ('preprocess', preprocessor),
    ('clf', XGBClassifier(
        n_estimators = 400,
        learning_rate = 0.01,
        max_depth = 4,
        scale_pos_weight = scale_pos_weight,
        eval_metric = 'logloss',
        random_state = 42
    ))
])

# Train model
#------------

model.fit(X_train, y_train)

# Evaluation
#-----------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print('---Classification Report---')
print(classification_report(y_test, y_pred))

print('---Confusion Matrix---')
print(confusion_matrix(y_test, y_pred))

print('---ROC-AUC Score---')
print(roc_auc_score(y_test, y_prob))

# Threshold Tuning
# ------------------

threshold = 0.35
y_pred_adj = (y_prob > threshold).astype(int)

print("\n--- Classification Report (Threshold = 0.35) ---")
print(classification_report(y_test, y_pred_adj))

# Save model
# --------------

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_xgb.pkl")

print("\nModel training complete!")
print("Saved final model to: models/model_xgb.pkl")