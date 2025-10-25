"""
Phishing URL Detection - Model Training Script

This script extracts features from URLs and trains a Random Forest classifier
to detect phishing websites.

Features extracted:
- URL length
- Number of dots, hyphens, underscores, slashes
- Number of digits
- Presence of '@' symbol
- Presence of IP address
- Subdomain count
- Path length, query length
- Number of special characters
- HTTPS presence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os


def load_dataset(filepath='data/dataset.csv'):
    """
    Load the phishing URL dataset
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame containing the dataset
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Phishing URLs: {len(df[df['label'] == 1])}")
    print(f"Legitimate URLs: {len(df[df['label'] == 0])}")
    return df


def prepare_data(df):
    """
    Prepare features and labels for training
    
    Args:
        df: DataFrame containing the dataset
    
    Returns:
        X: Feature matrix
        y: Labels
    """
    # Select feature columns (all columns except 'url' and 'label')
    feature_columns = [col for col in df.columns if col not in ['url', 'label']]
    
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"\nFeatures used: {feature_columns}")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y, feature_columns


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    print("\nTraining Random Forest model...")
    
    # Initialize Random Forest with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("="*50)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Legit  Phishing")
    print(f"Actual Legit    {cm[0][0]:3d}     {cm[0][1]:3d}")
    print(f"    Phishing    {cm[1][0]:3d}     {cm[1][1]:3d}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Feature importance
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def save_model(model, feature_columns, filepath='model/phishing_model.pkl'):
    """
    Save the trained model to disk
    
    Args:
        model: Trained model
        feature_columns: List of feature names
        filepath: Path to save the model
    """
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model and feature columns together
    model_data = {
        'model': model,
        'feature_columns': feature_columns
    }
    
    joblib.dump(model_data, filepath)
    print(f"\nModel saved to {filepath}")


def display_feature_importance(model, feature_columns):
    """
    Display feature importance from the trained model
    
    Args:
        model: Trained Random Forest model
        feature_columns: List of feature names
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    for idx, row in feature_importance_df.iterrows():
        print(f"{row['Feature']:20s}: {row['Importance']:.4f}")
    print("="*50)


def main():
    """
    Main function to orchestrate the training process
    """
    print("\n" + "="*60)
    print("PHISHING URL DETECTION - MODEL TRAINING")
    print("="*60 + "\n")
    
    # Step 1: Load dataset
    df = load_dataset('data/dataset.csv')
    
    # Step 2: Prepare data
    X, y, feature_columns = prepare_data(df)
    
    # Step 3: Split data into training and testing sets
    print("\nSplitting data: 80% training, 20% testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 6: Display feature importance
    display_feature_importance(model, feature_columns)
    
    # Step 7: Save model
    save_model(model, feature_columns, 'model/phishing_model.pkl')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")
    
    return model, metrics


if __name__ == "__main__":
    main()
