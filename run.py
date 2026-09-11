"""
run.py
Convenience runner to train models or launch the Flask application.

Usage:
    python run.py            # Launch Flask application (auto-trains if model missing)
    python run.py --train    # Force retrain models and update metadata
"""

import sys
import os

# Ensure src/ is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == '__main__':
    if '--train' in sys.argv or '-t' in sys.argv:
        print("[*] Starting model training pipeline...")
        from train import train_and_evaluate
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(base_dir, "data", "telco_churn.csv")
        models_folder = os.path.join(base_dir, "models")
        train_and_evaluate(data_file, models_folder)
        print("[*] Training complete!")
    else:
        # Launch Flask app
        from app import app
        port = int(os.environ.get('PORT', 5000))
        print(f"\n========================================================")
        print(f"🚀 Telco Customer Churn Prediction Platform")
        print(f"🌐 Web App URL:  http://127.0.0.1:{port}/")
        print(f"📊 Metrics URL:  http://127.0.0.1:{port}/metrics")
        print(f"🔌 API Endpoint: http://127.0.0.1:{port}/api/predict")
        print(f"========================================================\n")
        app.run(host='0.0.0.0', port=port, debug=False)
