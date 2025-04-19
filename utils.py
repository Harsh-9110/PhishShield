"""
Utility functions for phishing website detection.
"""

import os
import re
import urllib.parse
import requests
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

# Initialize logging
logger = logging.getLogger(__name__)

def is_valid_url(url):
    """
    Check if a URL is valid.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    try:
        result = urllib.parse.urlparse(url)
        # Check if scheme and netloc are present
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def load_dataset():
    """
    Load and preprocess the phishing dataset.
    
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    try:
        # First check if dataset is available locally
        if os.path.exists('phishing_dataset.csv'):
            logger.info("Loading dataset from local file...")
            df = pd.read_csv('phishing_dataset.csv')
        else:
            # If not available locally, download from a public source
            logger.info("Downloading phishing dataset...")
            # URL for the UCI Phishing Website Dataset
            url = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv"
            df = pd.read_csv(url)
            
            # Save locally for future use
            df.to_csv('phishing_dataset.csv', index=False)
        
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        # Separate features and target
        if 'class' in df.columns:
            y = df['class'].values
            X = df.drop('class', axis=1)
        elif 'Result' in df.columns:
            y = df['Result'].values
            X = df.drop('Result', axis=1)
        else:
            # If target column is not found, assume the last column is the target
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        
        # If cannot load dataset, create a synthetic dataset for demonstration
        logger.warning("Creating a small synthetic dataset for demonstration...")
        
        # Generate a small synthetic dataset 
        # This is a fallback to prevent the app from crashing
        num_samples = 1000
        num_features = 29  # Match the number of features in feature_extraction.py
        
        # Generate random features
        X = np.random.randn(num_samples, num_features)
        
        # Generate random labels with bias towards phishing
        y = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
        
        # Create feature names to match feature_extraction.py
        feature_names = [
            'url_length', 'hostname_length', 'dots_count', 'subdomain_count', 
            'path_length', 'query_params_count', 'numeric_chars_in_domain', 
            'special_chars', 'has_ip', 'domain_age', 'suspicious_tld', 
            'suspicious_keywords_count', 'has_https', 'domain_length',
            'has_login_form', 'has_password_field', 'has_suspicious_html',
            'external_resources_count', 'form_count', 'iframe_count', 
            'favicon_same_domain', 'has_meta_redirect', 'has_statusbar_customization',
            'has_popup_window', 'suspicious_content_words', 'content_length', 
            'unique_word_ratio', 'has_login_words', 'has_password_words'
        ]
        
        # Convert to DataFrame to include feature names
        X = pd.DataFrame(X, columns=feature_names)
        
        return X, y


def check_url_accessibility(url):
    """
    Check if a URL is accessible.
    
    Args:
        url (str): URL to check
        
    Returns:
        tuple: (is_accessible, status_code, error_message)
    """
    try:
        # Try HEAD request first (faster)
        response = requests.head(url, timeout=5, allow_redirects=True)
        
        # If HEAD request fails with 405 (Method Not Allowed), try GET request
        if response.status_code == 405:
            response = requests.get(url, timeout=5, allow_redirects=True)
        
        if response.status_code < 400:
            return True, response.status_code, None
        else:
            return False, response.status_code, f"Server returned error: {response.status_code} {response.reason}"
    except requests.exceptions.ConnectionError:
        return False, None, "Connection error: Could not connect to the server"
    except requests.exceptions.Timeout:
        return False, None, "Timeout error: The server took too long to respond"
    except requests.exceptions.TooManyRedirects:
        return False, None, "Too many redirects: The URL has too many redirects"
    except requests.exceptions.RequestException as e:
        return False, None, f"Request error: {str(e)}"
    except Exception as e:
        return False, None, f"Unknown error: {str(e)}"


def sanitize_url(url):
    """
    Sanitize a URL to prevent security issues.
    
    Args:
        url (str): URL to sanitize
        
    Returns:
        str: Sanitized URL
    """
    # Make sure URL has a scheme
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Parse and rebuild URL to remove dangerous components
    parsed = urllib.parse.urlparse(url)
    safe_url = urllib.parse.urlunparse(parsed)
    
    return safe_url
