# PhishGuard - Phishing Website Detection Tool

PhishGuard is an advanced phishing website detection tool that uses machine learning and natural language processing to identify potential phishing threats. It analyzes URLs and website content to determine if a website is likely legitimate or a phishing attempt.

## Features

- **Multiple ML Models**: Uses Decision Tree, Random Forest, and XGBoost models for accurate phishing detection
- **Feature Extraction**: Extracts meaningful features from URLs and website content
- **Content Analysis**: Uses NLP to analyze the text content of websites
- **Visual Explanations**: Provides visualizations of feature importance
- **History Tracking**: Stores analysis results in a database for future reference
- **Browser Extension**: Integrates with your browser to analyze websites in real-time

## Technology Stack

- **Backend**: Python, Flask
- **Database**: PostgreSQL
- **Machine Learning**: scikit-learn, XGBoost
- **NLP**: NLTK, Trafilatura
- **Frontend**: Bootstrap, Chart.js
- **Deployment**: Vercel

## How It Works

1. A user submits a URL to be analyzed
2. The application extracts features from the URL and its content
3. Multiple machine learning models analyze these features
4. Results are aggregated and displayed with confidence scores
5. The analysis is stored in the database for future reference

## Local Development

To run this application locally:

1. Clone the repository
2. Install Python 3.9 or higher
3. Install dependencies: `pip install -r requirements-vercel.txt`
4. Set up a PostgreSQL database and configure the DATABASE_URL environment variable
5. Run: `python main.py`

## Browser Extension

The PhishGuard browser extension provides real-time phishing detection right in your browser:

1. Check any website with a single click
2. View detailed analysis from multiple ML models
3. Keep track of your browsing history
4. Configure settings to suit your needs

To install the extension:
1. Navigate to the `extension` directory
2. Follow the instructions in `README.md`

## API Integration

PhishGuard provides a REST API for integrating with other applications:

- **Endpoint**: `/api/check`
- **Method**: POST
- **Request Body**: JSON with a `url` parameter
- **Response**: Analysis results including phishing risk and confidence scores

For more details, see `extension/API_TESTING.md`.

## Deployment

This application is configured for deployment on Vercel. To deploy it:

1. Fork this repository
2. Connect to your Vercel account
3. Add required environment variables (DATABASE_URL, etc.)
4. Deploy!

---
Code By Harsh