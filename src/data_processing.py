"""
data_processing.py
Data ingestion, cleaning, and preprocessing pipeline for Telco Churn Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

CATEGORICAL_FEATURES = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

TARGET_COL = 'Churn'


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw Telco customer churn dataset from CSV."""
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Telco Customer dataset:
    - Drop customerID if exists
    - Convert TotalCharges to numeric (handles blanks/spaces)
    - Impute TotalCharges missing values (tenure==0 defaults to MonthlyCharges or 0)
    - Convert target Churn ('Yes'/'No') to 1/0 if present
    """
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # TotalCharges contains whitespace characters in some rows
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        # If tenure is 0 or TotalCharges is NaN, fill TotalCharges with MonthlyCharges or 0
        if 'MonthlyCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
        else:
            df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # Convert SeniorCitizen to categorical string for consistent OneHot encoding
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    # Encode target column if present
    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype == object:
            df[TARGET_COL] = df[TARGET_COL].map({'Yes': 1, 'No': 0})

    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Construct the ColumnTransformer pipeline:
    - Numeric: Median Imputer + StandardScaler
    - Categorical: OneHotEncoder with handle_unknown='ignore'
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def prepare_input_df(data: dict) -> pd.DataFrame:
    """
    Convert an input dictionary (from web form or API JSON) into
    a sanitized 1-row pandas DataFrame matching training schema.
    """
    sample = {}

    # Numeric conversions
    try:
        sample['tenure'] = [float(data.get('tenure', 1))]
    except (ValueError, TypeError):
        sample['tenure'] = [1.0]

    try:
        sample['MonthlyCharges'] = [float(data.get('MonthlyCharges', 50.0))]
    except (ValueError, TypeError):
        sample['MonthlyCharges'] = [50.0]

    # If TotalCharges is not supplied or empty, estimate tenure * MonthlyCharges
    total_val = data.get('TotalCharges', '')
    if str(total_val).strip() == '' or total_val is None:
        sample['TotalCharges'] = [sample['tenure'][0] * sample['MonthlyCharges'][0]]
    else:
        try:
            sample['TotalCharges'] = [float(total_val)]
        except (ValueError, TypeError):
            sample['TotalCharges'] = [sample['tenure'][0] * sample['MonthlyCharges'][0]]

    # Categorical conversions with robust defaults
    defaults = {
        'gender': 'Female',
        'SeniorCitizen': '0',
        'Partner': 'No',
        'Dependents': 'No',
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
        'PaymentMethod': 'Electronic check'
    }

    for col in CATEGORICAL_FEATURES:
        val = data.get(col, defaults.get(col, 'No'))
        sample[col] = [str(val).strip()]

    df = pd.DataFrame(sample)
    return df
