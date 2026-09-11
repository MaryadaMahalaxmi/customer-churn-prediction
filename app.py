"""
app.py
Production Flask application for Telco Customer Churn Prediction.
Includes interactive web UI, prediction reporting, and RESTful API endpoints.
"""

import os
import sys

# Ensure src/ is on python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from flask import Flask, render_template, request, jsonify
from predict import get_predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'telco-churn-ml-secure-key-2026')


@app.route('/', methods=['GET'])
def index():
    """Render the main churn prediction form."""
    predictor = get_predictor()
    metadata = predictor.metadata.get('metrics', {})
    return render_template('index.html', metrics=metadata)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle web form submission and render prediction results."""
    try:
        form_data = request.form.to_dict()
        predictor = get_predictor()
        result = predictor.predict_single(form_data)
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    RESTful JSON API endpoint for external integrations.
    Expected JSON payload: dictionary of customer attributes.
    """
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Request must be JSON with Content-Type: application/json"
        }), 400

    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({
                "status": "error",
                "message": "Empty JSON payload received"
            }), 400

        predictor = get_predictor()
        result = predictor.predict_single(input_data)
        return jsonify({
            "status": "success",
            "data": result
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Display model performance evaluation, confusion matrix, and feature importances."""
    predictor = get_predictor()
    metadata = predictor.metadata
    return render_template('metrics.html', metadata=metadata)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring and uptime probes."""
    predictor = get_predictor()
    is_ready = predictor.pipeline is not None
    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "best_model": predictor.metadata.get('best_model', 'N/A')
    }), 200 if is_ready else 503


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    print(f"[*] Starting Telco Customer Churn Prediction Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
