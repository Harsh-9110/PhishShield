"""
Web scraper module for fetching and parsing website content.
"""

import requests
import logging
import trafilatura
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Initialize logging
logger = logging.getLogger(__name__)

def get_website_text_content(url, timeout=10):
    """
    Fetches and extracts the main text content of a website.
    
    Args:
        url (str): URL of the website to scrape
        timeout (int, optional): Request timeout in seconds. Defaults to 10.
        
    Returns:
        str: Extracted text content or None if failed
    """
    try:
        # Fetch the content using trafilatura (without timeout as it's not supported in all versions)
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            # Extract main text content
            text = trafilatura.extract(downloaded)
            return text
        
        # Fallback to requests + BeautifulSoup if trafilatura fails
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    except Exception as e:
        logger.error(f"Error scraping website {url}: {str(e)}")
        return None


def extract_html_features(url, timeout=10):
    """
    Extracts features from the HTML content of a website.
    
    Args:
        url (str): URL of the website
        timeout (int, optional): Request timeout in seconds. Defaults to 10.
        
    Returns:
        dict: Dictionary of HTML features
    """
    features = {
        'has_login_form': 0,
        'has_password_field': 0,
        'has_suspicious_html': 0,
        'external_resources_count': 0,
        'form_count': 0,
        'iframe_count': 0,
        'favicon_same_domain': 1,
        'has_meta_redirect': 0,
        'has_statusbar_customization': 0,
        'has_popup_window': 0
    }
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse domain
        domain = urlparse(url).netloc
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Count forms
        forms = soup.find_all('form')
        features['form_count'] = len(forms)
        
        # Check for login forms
        for form in forms:
            inputs = form.find_all('input')
            input_types = [inp.get('type', '').lower() for inp in inputs]
            
            if 'password' in input_types:
                features['has_password_field'] = 1
                
            if ('password' in input_types and 
                ('text' in input_types or 'email' in input_types or 'tel' in input_types)):
                features['has_login_form'] = 1
        
        # Count iframes
        iframes = soup.find_all('iframe')
        features['iframe_count'] = len(iframes)
        
        # Check external resources
        ext_resources = 0
        for tag in soup.find_all(['script', 'img', 'link']):
            src = tag.get('src', tag.get('href', ''))
            if src and src.startswith('http') and domain not in src:
                ext_resources += 1
        
        features['external_resources_count'] = ext_resources
        
        # Check for meta redirects
        if soup.find('meta', attrs={'http-equiv': 'refresh'}):
            features['has_meta_redirect'] = 1
        
        # Check for favicon
        favicon = soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')
        if favicon and favicon.get('href', '').startswith('http') and domain not in favicon['href']:
            features['favicon_same_domain'] = 0
        
        # Check for suspicious JavaScript
        scripts = soup.find_all('script')
        for script in scripts:
            script_content = script.string or ''
            
            # Check for status bar customization
            if 'window.status' in script_content or 'status=' in script_content:
                features['has_statusbar_customization'] = 1
            
            # Check for popup windows
            if ('window.open' in script_content or 'open(' in script_content or 
                'popup' in script_content.lower()):
                features['has_popup_window'] = 1
            
            # Check for suspicious redirects or obfuscation
            if ('document.location' in script_content or 'window.location' in script_content or
                'eval(' in script_content or 'unescape(' in script_content):
                features['has_suspicious_html'] = 1
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting HTML features from {url}: {str(e)}")
        return features
