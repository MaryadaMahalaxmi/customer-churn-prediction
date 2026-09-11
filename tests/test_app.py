"""
test_app.py
Integration tests for Flask web routes and RESTful API endpoints.
"""

import os
import sys
import json
import unittest

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


class TestFlaskRoutes(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        """Verify home page loads properly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Telco', response.data)
        self.assertIn(b'churnForm', response.data)

    def test_health_route(self):
        """Verify health check endpoint returns 200."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)

    def test_metrics_route(self):
        """Verify metrics dashboard returns 200."""
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Model Performance', response.data)

    def test_form_predict_route(self):
        """Verify web form prediction returns 200 with result view."""
        form_payload = {
            'gender': 'Male',
            'SeniorCitizen': '0',
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': '36',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': '55.00',
            'TotalCharges': '1980.00'
        }
        response = self.client.post('/predict', data=form_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction Result', response.data)
        self.assertIn(b'Churn Probability', response.data)

    def test_api_predict_valid_json(self):
        """Verify REST API endpoint processes valid JSON correctly."""
        json_payload = {
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.70,
            "TotalCharges": 151.65
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(json_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        self.assertEqual(res_data['status'], 'success')
        self.assertIn('data', res_data)
        self.assertIn('churn_prediction', res_data['data'])
        self.assertIn('churn_probability', res_data['data'])
        self.assertIn('risk_level', res_data['data'])

    def test_api_predict_non_json_rejection(self):
        """Verify non-JSON request is rejected with HTTP 400."""
        response = self.client.post(
            '/api/predict',
            data="Not a json string",
            content_type='text/plain'
        )
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
