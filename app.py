"""
Phishing URL Detection - Flask Web Application

This Flask application provides a web interface for detecting phishing URLs.
Users can input a URL and get instant predictions.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import re
from urllib.parse import urlparse
import os

app = Flask(__name__)

# Global variable to store the loaded model
MODEL = None
FEATURE_COLUMNS = None


def load_model():
    """
    Load the trained model from disk
    """
    global MODEL, FEATURE_COLUMNS
    
    model_path = 'model/phishing_model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please run 'python model/train_model.py' first to train the model."
        )
    
    model_data = joblib.load(model_path)
    MODEL = model_data['model']
    FEATURE_COLUMNS = model_data['feature_columns']
    
    print(f"Model loaded successfully!")
    print(f"Features: {FEATURE_COLUMNS}")


def extract_features(url):
    """
    Extract features from a given URL
    
    Args:
        url: The URL string to analyze
    
    Returns:
        Dictionary containing extracted features
    """
    features = {}
    
    # Basic URL metrics
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_digits'] = sum(c.isdigit() for c in url)
    
    # Check for '@' symbol (often used in phishing)
    features['has_at'] = 1 if '@' in url else 0
    
    # Check if URL contains IP address
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    features['has_ip'] = 1 if re.search(ip_pattern, url) else 0
    
    # Parse URL components
    try:
        parsed = urlparse(url)
        
        # Count subdomains
        domain = parsed.netloc
        if domain:
            # Remove port if present
            domain = domain.split(':')[0]
            # Count dots in domain (subdomains)
            features['subdomain_count'] = domain.count('.')
        else:
            features['subdomain_count'] = 0
        
        # Path length
        features['path_length'] = len(parsed.path) if parsed.path else 0
        
        # Query length
        features['query_length'] = len(parsed.query) if parsed.query else 0
        
        # HTTPS check
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        
    except Exception as e:
        print(f"Error parsing URL: {e}")
        features['subdomain_count'] = 0
        features['path_length'] = 0
        features['query_length'] = 0
        features['has_https'] = 0
    
    # Count special characters (excluding common ones in URLs)
    special_chars = r'[!#$%&*+=?^`{|}~]'
    features['num_special_chars'] = len(re.findall(special_chars, url))
    
    return features


def predict_url(url):
    """
    Predict if a URL is phishing or legitimate
    
    Args:
        url: The URL to analyze
    
    Returns:
        Dictionary containing prediction results
    """
    # Extract features
    features = extract_features(url)
    
    # Create feature vector in the correct order
    feature_vector = [features[col] for col in FEATURE_COLUMNS]
    
    # Make prediction
    prediction = MODEL.predict([feature_vector])[0]
    probability = MODEL.predict_proba([feature_vector])[0]
    
    # Prepare result
    result = {
        'url': url,
        'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
        'is_phishing': bool(prediction),
        'confidence': float(max(probability)) * 100,
        'phishing_probability': float(probability[1]) * 100,
        'legitimate_probability': float(probability[0]) * 100,
        'features': features
    }
    
    return result


@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for URL prediction
    """
    try:
        # Get URL from request
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'error': 'Please provide a URL'
            }), 400
        
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Make prediction
        result = predict_url(url)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None
    })


@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors
    """
    return render_template('index.html'), 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PHISHING URL DETECTION - WEB APPLICATION")
    print("="*60 + "\n")
    
    # Load the model
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease train the model first by running:")
        print("  python model/train_model.py")
        exit(1)
    
    print("\nStarting Flask web server...")
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
