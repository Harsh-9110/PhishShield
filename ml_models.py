"""
ML Models module for phishing website detection.
Implements multiple models and provides prediction logic.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize logging
logger = logging.getLogger(__name__)

def train_models(X, y):
    """
    Train multiple ML models on the provided dataset.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target vector (0 for legitimate, 1 for phishing)
        
    Returns:
        dict: Dictionary containing trained models and evaluation metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    xgb_model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42)
    
    # Train models
    logger.info("Training Decision Tree model...")
    dt_model.fit(X_train, y_train)
    
    logger.info("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    logger.info("Training XGBoost model...")
    xgb_model.fit(X_train, y_train)
    
    # Evaluate models
    models_metrics = {}
    models = {
        'decision_tree': dt_model,
        'random_forest': rf_model,
        'xgboost': xgb_model
    }
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        models_metrics[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Return trained models and metrics
    return {
        'models': models,
        'metrics': models_metrics,
        'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None
    }


def predict_phishing(models, features):
    """
    Predict if a URL is phishing using multiple models.
    
    Args:
        models (dict): Dictionary of trained models
        features (array): Feature array for the URL
        
    Returns:
        dict: Prediction results from all models
    """
    results = {}
    
    for name, model in models.items():
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability/confidence
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(features)[0]
            confidence = probas[1] if prediction == 1 else probas[0]
        else:
            # XGBoost may use different function
            probas = model.predict_proba(features)[0]
            confidence = probas[1] if prediction == 1 else probas[0]
        
        results[name] = {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'is_phishing': bool(prediction == 1)
        }
    
    return results
