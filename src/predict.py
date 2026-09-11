"""
predict.py
Inference engine and risk-scoring wrapper for Telco Customer Churn.
"""

import os
import json
import joblib
import pandas as pd

from data_processing import prepare_input_df


class ChurnPredictor:
    """
    Production-grade churn predictor that encapsulates pipeline loading,
    probabilistic scoring, risk stratification, and business retention rules.
    """

    def __init__(self, model_path=None, metadata_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(base_dir, "models", "churn_pipeline.pkl")
        self.metadata_path = metadata_path or os.path.join(base_dir, "models", "model_metadata.json")

        self.pipeline = None
        self.metadata = {}
        self._load_model()

    def _load_model(self):
        """Load trained pipeline and metadata; train on-the-fly if not found."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                print(f"[!] Error loading {self.model_path}: {e}")

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"[!] Error loading {self.metadata_path}: {e}")

        # If model is not generated yet, trigger training automatically
        if self.pipeline is None:
            print("[*] Model pipeline not found. Initializing training...")
            try:
                from train import train_and_evaluate
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                data_path = os.path.join(base_dir, "data", "telco_churn.csv")
                output_dir = os.path.join(base_dir, "models")
                self.pipeline, self.metadata = train_and_evaluate(data_path, output_dir)
            except Exception as e:
                print(f"[!] Fallback training failed: {e}")

    def predict_single(self, input_dict: dict) -> dict:
        """
        Execute prediction for a single customer payload.
        Returns prediction, probability, risk tier, risk drivers, and retention actions.
        """
        if self.pipeline is None:
            self._load_model()

        df_input = prepare_input_df(input_dict)

        if self.pipeline is not None:
            proba_array = self.pipeline.predict_proba(df_input)[0]
            churn_prob = float(proba_array[1])
            prediction = "Yes" if churn_prob >= 0.5 else "No"
        else:
            # Heuristic fallback if pipeline is unpickled in restricted environment
            tenure = float(df_input['tenure'].iloc[0])
            contract = df_input['Contract'].iloc[0]
            monthly = float(df_input['MonthlyCharges'].iloc[0])
            score = 0.5
            if contract == 'Month-to-month': score += 0.25
            if contract == 'Two year': score -= 0.35
            if tenure < 6: score += 0.2
            if monthly > 80: score += 0.1
            churn_prob = min(max(score, 0.05), 0.95)
            prediction = "Yes" if churn_prob >= 0.5 else "No"

        churn_percentage = round(churn_prob * 100, 1)

        # Determine Risk Level Tier
        if churn_percentage >= 65.0:
            risk_level = "High"
            risk_badge_class = "badge-high"
        elif churn_percentage >= 35.0:
            risk_level = "Medium"
            risk_badge_class = "badge-medium"
        else:
            risk_level = "Low"
            risk_badge_class = "badge-low"

        # Diagnostic Risk Factors
        risk_factors = []
        contract = df_input['Contract'].iloc[0]
        tenure = float(df_input['tenure'].iloc[0])
        payment_method = df_input['PaymentMethod'].iloc[0]
        monthly_charges = float(df_input['MonthlyCharges'].iloc[0])
        tech_support = df_input['TechSupport'].iloc[0]
        online_security = df_input['OnlineSecurity'].iloc[0]
        internet = df_input['InternetService'].iloc[0]

        if contract == 'Month-to-month':
            risk_factors.append("Month-to-month contract offers zero switching barrier.")
        if tenure <= 12:
            risk_factors.append(f"Early customer lifecycle tenure ({int(tenure)} months).")
        if payment_method == 'Electronic check':
            risk_factors.append("Electronic check payment is historically correlated with elevated churn.")
        if monthly_charges >= 75.0:
            risk_factors.append(f"High monthly bill (${monthly_charges:.2f}/mo) increases price sensitivity.")
        if tech_support == 'No' and internet != 'No':
            risk_factors.append("Lacks Tech Support subscription for unresolved service friction.")
        if online_security == 'No' and internet != 'No':
            risk_factors.append("No Online Security add-on weakens customer lock-in.")

        if not risk_factors:
            risk_factors.append("Strong tenure and contract commitment keep risk indicators low.")

        # Tailored Business Retention Recommendations
        recommendations = []
        if contract == 'Month-to-month':
            recommendations.append("Offer a 15% discount for upgrading to a 1-year or 2-year agreement.")
        if payment_method == 'Electronic check':
            recommendations.append("Incentivize automatic credit card / ACH billing with a $10 one-time bill credit.")
        if tech_support == 'No' or online_security == 'No':
            recommendations.append("Offer a complimentary 3-month trial of Premium Security & Tech Support bundle.")
        if tenure <= 6:
            recommendations.append("Schedule a dedicated proactive CS onboarding check-in call.")
        if monthly_charges > 85.0:
            recommendations.append("Review plan options to tailor a customized value bundle.")
        if not recommendations:
            recommendations.append("Enroll customer in VIP loyalty perks & early-access upgrade programs.")

        return {
            "churn_prediction": prediction,
            "churn_probability": churn_percentage,
            "risk_level": risk_level,
            "risk_badge_class": risk_badge_class,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "customer_profile": df_input.iloc[0].to_dict()
        }


# Global singleton predictor instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
