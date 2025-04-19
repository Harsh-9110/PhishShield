import os
import logging
import urllib.parse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS

# Import our custom modules
from feature_extraction import extract_features_from_url
from ml_models import train_models, predict_phishing
from web_scraper import get_website_text_content
from utils import is_valid_url, load_dataset, check_url_accessibility

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

# Initialize the app and db
db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
# Enable CORS for the API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.secret_key = os.environ.get("SESSION_SECRET", "phishing_detector_secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
database_url = os.environ.get("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    # Heroku-style URI needs to be updated for SQLAlchemy 1.4+
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url or "sqlite:///phishing.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models
    import models
    # Create database tables
    db.create_all()
    
    # Load and prepare dataset, train models if not already loaded
    if not hasattr(app, 'models_trained'):
        try:
            # Load dataset and train models
            X, y = load_dataset()
            
            if X is not None and y is not None:
                model_data = train_models(X, y)
                app.models = model_data['models']
                app.feature_names = model_data['feature_names']
                app.models_trained = True
                logger.info("ML models trained successfully.")
            else:
                logger.warning("Could not load dataset. Models not trained.")
                app.models_trained = False
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            app.models_trained = False


@app.route('/')
def index():
    return render_template('index.html', models_trained=getattr(app, 'models_trained', False))


@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form.get('url', '')
    
    if not url:
        flash('Please enter a URL', 'danger')
        return redirect(url_for('index'))
    
    # Make sure URL has a scheme
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Validate URL
    if not is_valid_url(url):
        flash('Invalid URL format', 'danger')
        return redirect(url_for('index'))
    
    # Check if URL is accessible
    is_accessible, status_code, error_message = check_url_accessibility(url)
    if not is_accessible:
        if status_code:
            flash(f'URL is not accessible. {error_message}', 'danger')
        else:
            flash(f'URL is not accessible. {error_message}', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Extract features from the URL
        features, feature_dict = extract_features_from_url(url)
        
        # Only continue if models are trained
        if getattr(app, 'models_trained', False):
            # Get predictions from all models
            results = predict_phishing(app.models, features)
            
            # Create feature importance visualization
            feature_importance_plot = create_feature_importance_plot(feature_dict)
            
            # Save results to session
            session['results'] = results
            session['feature_dict'] = feature_dict
            session['url'] = url
            session['feature_importance_plot'] = feature_importance_plot
            
            # Add check result to database
            try:
                check_result = models.CheckResult(
                    url=url,
                    dt_prediction=results['decision_tree']['prediction'],
                    rf_prediction=results['random_forest']['prediction'],
                    xgb_prediction=results['xgboost']['prediction'],
                    dt_confidence=results['decision_tree']['confidence'],
                    rf_confidence=results['random_forest']['confidence'],
                    xgb_confidence=results['xgboost']['confidence']
                )
                db.session.add(check_result)
                db.session.commit()
            except Exception as e:
                logger.error(f"Error saving result to database: {str(e)}")
            
            return redirect(url_for('results'))
        else:
            flash('Models are not trained. Cannot perform analysis.', 'warning')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error analyzing URL: {str(e)}")
        flash(f'Error analyzing URL: The website could not be analyzed. It may be inaccessible or not support the required features.', 'danger')
        return redirect(url_for('index'))


@app.route('/results')
def results():
    if 'results' not in session:
        flash('No analysis results available', 'warning')
        return redirect(url_for('index'))
    
    results = session['results']
    feature_dict = session['feature_dict']
    url = session['url']
    feature_importance_plot = session.get('feature_importance_plot', '')
    
    # Determine overall result based on majority voting
    predictions = [
        results['decision_tree']['prediction'],
        results['random_forest']['prediction'],
        results['xgboost']['prediction']
    ]
    is_phishing = predictions.count(1) > predictions.count(0)
    
    # Get top 5 features that contributed to prediction
    top_features = sorted(
        feature_dict.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:5]
    
    return render_template(
        'results.html',
        url=url,
        results=results,
        is_phishing=is_phishing,
        feature_dict=feature_dict,
        top_features=top_features,
        feature_importance_plot=feature_importance_plot
    )


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/history')
def history():
    # Get the most recent 10 check results from the database
    check_results = models.CheckResult.query.order_by(models.CheckResult.timestamp.desc()).limit(10).all()
    return render_template('history.html', check_results=check_results)


@app.route('/api/check', methods=['POST'])
def api_check():
    """API endpoint for the browser extension to check URLs"""
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return {'error': 'No URL provided'}, 400
    
    # Make sure URL has a scheme
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Validate URL
    if not is_valid_url(url):
        return {'error': 'Invalid URL format'}, 400
    
    # Check if URL is accessible
    is_accessible, status_code, error_message = check_url_accessibility(url)
    if not is_accessible:
        return {'error': f'URL is not accessible: {error_message}'}, 400
    
    try:
        # Extract features from the URL
        features, feature_dict = extract_features_from_url(url)
        
        # Only continue if models are trained
        if getattr(app, 'models_trained', False):
            # Get predictions from all models
            results = predict_phishing(app.models, features)
            
            # Add check result to database
            try:
                check_result = models.CheckResult(
                    url=url,
                    dt_prediction=results['decision_tree']['prediction'],
                    rf_prediction=results['random_forest']['prediction'],
                    xgb_prediction=results['xgboost']['prediction'],
                    dt_confidence=results['decision_tree']['confidence'],
                    rf_confidence=results['random_forest']['confidence'],
                    xgb_confidence=results['xgboost']['confidence']
                )
                db.session.add(check_result)
                db.session.commit()
            except Exception as e:
                logger.error(f"Error saving result to database: {str(e)}")
            
            # Determine overall result based on majority voting
            predictions = [
                results['decision_tree']['prediction'],
                results['random_forest']['prediction'],
                results['xgboost']['prediction']
            ]
            is_phishing = predictions.count(1) > predictions.count(0)
            
            # Return the results
            return {
                'url': url,
                'is_phishing': bool(is_phishing),
                'confidence': {
                    'decision_tree': results['decision_tree']['confidence'],
                    'random_forest': results['random_forest']['confidence'],
                    'xgboost': results['xgboost']['confidence']
                },
                'predictions': {
                    'decision_tree': results['decision_tree']['prediction'],
                    'random_forest': results['random_forest']['prediction'],
                    'xgboost': results['xgboost']['prediction']
                }
            }
        else:
            return {'error': 'Models are not trained'}, 500
            
    except Exception as e:
        logger.error(f"Error analyzing URL: {str(e)}")
        return {'error': 'Error analyzing URL'}, 500


def create_feature_importance_plot(feature_dict):
    """Create a feature importance visualization"""
    try:
        # Create a bar chart of the feature values
        plt.figure(figsize=(10, 6))
        
        # Select top 10 features by absolute value
        top_features = sorted(
            feature_dict.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]
        
        # Convert to dataframe for plotting
        df = pd.DataFrame(top_features, columns=['Feature', 'Value'])
        
        # Create bar chart
        colors = ['#ff7675' if x < 0 else '#74b9ff' for x in df['Value']]
        sns.barplot(x='Value', y='Feature', data=df, palette=colors)
        plt.title('Top 10 Features by Importance')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Encode to base64 string
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        return ""
