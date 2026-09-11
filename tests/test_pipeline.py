"""
test_pipeline.py
Unit tests for data cleaning, preprocessing pipeline, and inference predictor.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure src/ is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data_processing import clean_data, get_preprocessor, prepare_input_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from predict import ChurnPredictor


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame({
            'customerID': ['0001-ABCD', '0002-EFGH', '0003-IJKL'],
            'gender': ['Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes'],
            'tenure': [1, 34, 0],
            'PhoneService': ['No', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['No', 'Yes', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'No internet service'],
            'DeviceProtection': ['No', 'Yes', 'No internet service'],
            'TechSupport': ['No', 'No', 'No internet service'],
            'StreamingTV': ['No', 'No', 'No internet service'],
            'StreamingMovies': ['No', 'No', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
            'MonthlyCharges': [29.85, 56.95, 20.0],
            'TotalCharges': ['29.85', '1889.5', ' '],  # Notice blank space in row 3
            'Churn': ['No', 'No', 'Yes']
        })

    def test_clean_data_removes_customer_id(self):
        cleaned = clean_data(self.sample_data)
        self.assertNotIn('customerID', cleaned.columns)

    def test_clean_data_handles_blank_total_charges(self):
        cleaned = clean_data(self.sample_data)
        # Check that TotalCharges is float and contains no NaN
        self.assertTrue(pd.api.types.is_float_dtype(cleaned['TotalCharges']))
        self.assertFalse(cleaned['TotalCharges'].isnull().any())
        # Row 3 had tenure=0, MonthlyCharges=20.0, blank TotalCharges -> filled with 20.0
        self.assertEqual(cleaned['TotalCharges'].iloc[2], 20.0)

    def test_clean_data_maps_churn_to_binary(self):
        cleaned = clean_data(self.sample_data)
        self.assertListEqual(cleaned['Churn'].tolist(), [0, 0, 1])

    def test_preprocessor_transformation(self):
        cleaned = clean_data(self.sample_data)
        X = cleaned.drop(columns=['Churn'])
        preprocessor = get_preprocessor()
        transformed = preprocessor.fit_transform(X)
        self.assertIsInstance(transformed, np.ndarray)
        self.assertFalse(np.isnan(transformed).any())

    def test_prepare_input_df(self):
        input_dict = {
            'gender': 'Male',
            'SeniorCitizen': '0',
            'tenure': '12',
            'MonthlyCharges': '65.5',
            'TotalCharges': '786.0',
            'Contract': 'One year'
        }
        df = prepare_input_df(input_dict)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['tenure'].iloc[0], 12.0)
        self.assertEqual(df['Contract'].iloc[0], 'One year')
        # Check defaults are populated for unprovided fields
        self.assertEqual(df['PhoneService'].iloc[0], 'Yes')


class TestChurnPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = ChurnPredictor()

    def test_predict_single_schema(self):
        payload = {
            'gender': 'Female',
            'SeniorCitizen': '0',
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 75.50,
            'TotalCharges': 151.00
        }
        result = self.predictor.predict_single(payload)

        self.assertIn('churn_prediction', result)
        self.assertIn(result['churn_prediction'], ['Yes', 'No'])
        self.assertIn('churn_probability', result)
        self.assertGreaterEqual(result['churn_probability'], 0.0)
        self.assertLessEqual(result['churn_probability'], 100.0)
        self.assertIn('risk_level', result)
        self.assertIn(result['risk_level'], ['Low', 'Medium', 'High'])
        self.assertIsInstance(result['risk_factors'], list)
        self.assertIsInstance(result['recommendations'], list)


if __name__ == '__main__':
    unittest.main()
