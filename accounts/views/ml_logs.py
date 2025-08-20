# ml_models/ml_model.py
import os
import re
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from django.conf import settings

class MLModel:
    MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_models", "rf_supervised.joblib")

    @staticmethod
    def load_model():
        if not os.path.exists(MLModel.MODEL_PATH):
            raise FileNotFoundError("Model not trained. Please run training script first.")
        data = joblib.load(MLModel.MODEL_PATH)
        return data['model'], data['encoders'], data['feature_order']

    @staticmethod
    def _extract_countries(route):
        if pd.isna(route):
            return None, None
        matches = re.findall(r"\(([^)]+)\)", str(route))
        return (matches[0].strip(), matches[-1].strip()) if matches else (None, None)

    @staticmethod
    def detect_anomalies(file_path):
        import os
        from django.conf import settings

        df = (
            pd.read_excel(file_path, dtype=str)
            if file_path.lower().endswith(('.xls', '.xlsx'))
            else pd.read_csv(file_path, dtype=str)
        )
        df.columns = [c.strip() for c in df.columns]

        incoming_col = next((c for c in df.columns if "incoming" in c.lower()), None)
        outgoing_col = next((c for c in df.columns if "outgoing" in c.lower()), None)

        if incoming_col:
            df['origin1'], df['dest1'] = zip(*df[incoming_col].apply(MLModel._extract_countries))
        else:
            df['origin1'], df['dest1'] = None, None

        if outgoing_col:
            df['origin2'], df['dest2'] = zip(*df[outgoing_col].apply(MLModel._extract_countries))
        else:
            df['origin2'], df['dest2'] = None, None

        try:
            model, encoders, feature_order = MLModel.load_model()
        except FileNotFoundError:
            raise FileNotFoundError("Trained model file not found. Please train the model first.")

        for col in ['origin1', 'dest1', 'origin2', 'dest2']:
            df[col] = df[col].fillna("Unknown")
            le = encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
            df[col] = le.transform(df[col])

        df['passenger_age'] = pd.to_numeric(df.get('passenger_age', np.nan), errors='coerce').fillna(9999)
        df['stay_duration'] = pd.to_numeric(df.get('stay_duration', np.nan), errors='coerce').fillna(9999)

        features = df[feature_order].astype(float)
        df['anomaly_score'] = model.predict_proba(features)[:, 1]

        # Risk classification
        def classify_risk(score):
            if score >= 0.8:
                return "High"
            elif score >= 0.6:
                return "Medium"
            elif score >= 0.4:
                return "Low"
            else:
                return "Normal"

        df['status'] = df['anomaly_score'].apply(classify_risk)

        # Legacy compatibility column (1 = High risk, 0 otherwise)
        df['is_potential_drug_trafficker'] = (df['status'] == "High").astype(int)

        # Save all passengers with risk level
        anomalies_dir = os.path.join(settings.MEDIA_ROOT, "anomalies")
        os.makedirs(anomalies_dir, exist_ok=True)

        filename = os.path.basename(file_path)
        anomalies_filename = f"anomalies_{os.path.splitext(filename)[0]}.csv"
        anomalies_path = os.path.join(anomalies_dir, anomalies_filename)

        df.to_csv(anomalies_path, index=False)
        print(f"[MLModel] Full passenger list with risk saved to: {anomalies_path}")

        return df
