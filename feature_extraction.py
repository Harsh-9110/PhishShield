"""
Feature extraction module for phishing website detection.
Extracts features from URLs and webpage content.
"""

import re
import urllib.parse
import logging
import numpy as np
import tldextract
from web_scraper import get_website_text_content
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize logging
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK resources: {str(e)}")

# Suspicious keywords often found in phishing URLs
SUSPICIOUS_KEYWORDS = [
    'secure', 'account', 'update', 'banking', 'login', 'signin', 
    'verify', 'confirm', 'paypal', 'password', 'credit', 'card',
    'bank', 'ebay', 'amazon', 'apple', 'microsoft', 'netflix', 'gmail'
]

# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS = [
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'info', 'online', 'site',
    'work', 'top', 'bid', 'stream', 'date', 'review', 'win'
]

def extract_features_from_url(url):
    """
    Extract features from a URL and its webpage content.
    
    Args:
        url (str): URL to analyze
        
    Returns:
        tuple: (feature_array, feature_dict) where feature_array is a numpy array 
               and feature_dict is a dictionary with feature names and values
    """
    feature_dict = {}
    
    # Parse URL
    parsed_url = urllib.parse.urlparse(url)
    domain_info = tldextract.extract(url)
    
    # 1. URL-based features
    
    # Length of URL
    url_length = len(url)
    feature_dict['url_length'] = url_length
    
    # Length of hostname
    hostname_length = len(parsed_url.netloc)
    feature_dict['hostname_length'] = hostname_length
    
    # Number of dots in URL
    dots_count = url.count('.')
    feature_dict['dots_count'] = dots_count
    
    # Number of subdomains
    subdomain_count = len(domain_info.subdomain.split('.')) if domain_info.subdomain else 0
    feature_dict['subdomain_count'] = subdomain_count
    
    # Path length
    path_length = len(parsed_url.path)
    feature_dict['path_length'] = path_length
    
    # Query parameters count
    query_params_count = len(urllib.parse.parse_qs(parsed_url.query))
    feature_dict['query_params_count'] = query_params_count
    
    # Number of numeric characters in domain
    numeric_chars = sum(c.isdigit() for c in parsed_url.netloc)
    feature_dict['numeric_chars_in_domain'] = numeric_chars
    
    # Special characters in URL
    special_chars = sum(c in "~`!@#$%^&*()_-+={}[]|\\:;\"'<>,.?/" for c in url)
    feature_dict['special_chars'] = special_chars
    
    # Has IP address in URL
    has_ip = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    feature_dict['has_ip'] = has_ip
    
    # Domain age (we can't calculate this without external APIs)
    # Set to -1 as a placeholder
    feature_dict['domain_age'] = -1
    
    # Check for suspicious TLD
    suspicious_tld = 1 if domain_info.suffix in SUSPICIOUS_TLDS else 0
    feature_dict['suspicious_tld'] = suspicious_tld
    
    # Check for suspicious keywords in domain
    suspicious_keywords_count = sum(keyword in domain_info.domain.lower() for keyword in SUSPICIOUS_KEYWORDS)
    feature_dict['suspicious_keywords_count'] = suspicious_keywords_count
    
    # Has 'https' in URL
    has_https = 1 if parsed_url.scheme == 'https' else 0
    feature_dict['has_https'] = has_https
    
    # Domain name length
    domain_length = len(domain_info.domain)
    feature_dict['domain_length'] = domain_length
    
    # 2. Content-based features
    # Default values in case we can't fetch the content
    feature_dict['has_login_form'] = 0
    feature_dict['has_password_field'] = 0
    feature_dict['has_suspicious_html'] = 0
    feature_dict['external_resources_count'] = 0
    feature_dict['form_count'] = 0
    feature_dict['iframe_count'] = 0
    feature_dict['favicon_same_domain'] = 1
    feature_dict['has_meta_redirect'] = 0
    feature_dict['has_statusbar_customization'] = 0
    feature_dict['has_popup_window'] = 0
    
    try:
        # Get webpage content
        content = get_website_text_content(url)
        
        if content:
            # Tokenize content
            tokens = word_tokenize(content.lower())
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            
            # Count suspicious words in content
            suspicious_content_words = sum(keyword in tokens for keyword in SUSPICIOUS_KEYWORDS)
            feature_dict['suspicious_content_words'] = suspicious_content_words
            
            # Content length
            content_length = len(content)
            feature_dict['content_length'] = content_length
            
            # Unique word ratio
            unique_words = len(set(tokens))
            unique_word_ratio = unique_words / len(tokens) if tokens else 0
            feature_dict['unique_word_ratio'] = unique_word_ratio
            
            # Check for login/password keywords
            has_login_words = 1 if any(word in tokens for word in ['login', 'sign', 'account', 'username']) else 0
            has_password_words = 1 if any(word in tokens for word in ['password', 'pwd', 'pass']) else 0
            feature_dict['has_login_words'] = has_login_words
            feature_dict['has_password_words'] = has_password_words
        else:
            # If couldn't get content, set defaults
            feature_dict['suspicious_content_words'] = -1
            feature_dict['content_length'] = -1
            feature_dict['unique_word_ratio'] = -1
            feature_dict['has_login_words'] = -1
            feature_dict['has_password_words'] = -1
    
    except Exception as e:
        logger.error(f"Error extracting content features: {str(e)}")
        # Set defaults if we couldn't get content
        feature_dict['suspicious_content_words'] = -1
        feature_dict['content_length'] = -1
        feature_dict['unique_word_ratio'] = -1
        feature_dict['has_login_words'] = -1
        feature_dict['has_password_words'] = -1
    
    # Convert feature dictionary to numpy array
    feature_names = list(feature_dict.keys())
    feature_array = np.array([feature_dict[name] for name in feature_names]).reshape(1, -1)
    
    return feature_array, feature_dict
