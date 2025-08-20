import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
historic_path = os.path.join(BASE_DIR, "media/train/historic_data.csv")
normal_path = os.path.join(BASE_DIR, "media/train/Emirates.csv")
model_save_path = os.path.join(BASE_DIR, "ml_models/rf_supervised.joblib")

df_anomalies = pd.read_csv(historic_path, dtype=str)
df_anomalies["is_anomaly"] = 1

df_normals = pd.read_csv(normal_path, dtype=str)
df_normals["is_anomaly"] = 0

df = pd.concat([df_anomalies, df_normals], ignore_index=True)

import re
def extract_countries(route):
    if pd.isna(route):
        return None, None
    matches = re.findall(r"\(([^)]+)\)", str(route))
    return (matches[0].strip(), matches[-1].strip()) if matches else (None, None)

incoming_col = next((c for c in df.columns if "incoming" in c.lower()), None)
outgoing_col = next((c for c in df.columns if "outgoing" in c.lower()), None)

df['origin1'], df['dest1'] = zip(*df[incoming_col].apply(extract_countries)) if incoming_col else (None, None)
df['origin2'], df['dest2'] = zip(*df[outgoing_col].apply(extract_countries)) if outgoing_col else (None, None)
encoders = {}
for col in ['origin1', 'dest1', 'origin2', 'dest2']:
    df[col] = df[col].fillna("Unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df['passenger_age'] = pd.to_numeric(df.get('passenger_age', np.nan), errors='coerce').fillna(9999)
df['stay_duration'] = pd.to_numeric(df.get('stay_duration', np.nan), errors='coerce').fillna(9999)

feature_order = ['origin1', 'dest1', 'origin2', 'dest2', 'passenger_age', 'stay_duration']
X = df[feature_order].astype(float)
y = df['is_anomaly'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_scores):.3f}")

# === SAVE MODEL ===
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump({'model': model, 'encoders': encoders, 'feature_order': feature_order}, model_save_path)
print(f"Model saved to {model_save_path}")
