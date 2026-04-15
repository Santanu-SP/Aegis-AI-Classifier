"""
Prediction Module for Spam Classifier.
Provides inference capabilities using the trained model and vectorizer.
"""

import os
import joblib
import numpy as np

# Adjust path for explicit import independent of run dir
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/spam_classifier.pkl')
VEC_PATH = os.path.join(os.path.dirname(__file__), '../models/tfidf_vectorizer.pkl')

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        raise FileNotFoundError("Model artifacts not found. Please run 'python src/train.py' first.")
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer

def predict(text: str) -> dict:
    """
    Predicts if a given text message is spam or ham.
    
    Args:
        text (str): The raw input message.
        
    Returns:
        dict: A dictionary containing:
            - classification: 'SPAM' or 'NOT SPAM'
            - confidence: float percentage 
            - top_spam_words: list of top 5 spam indicator words in the text
    """
    model, vectorizer = load_artifacts()
    
    # 1. Clean Text
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "classification": "NOT SPAM",
            "confidence": 100.0,
            "top_spam_words": [],
            "cleaned_text": ""
        }
    
    # 2. Extract Features
    X_vec = vectorizer.transform([cleaned])
    
    # 3. Predict
    pred = model.predict(X_vec)[0]
    prob = model.predict_proba(X_vec)[0]
    
    is_spam = bool(pred == 1)
    classification = "SPAM" if is_spam else "NOT SPAM"
    confidence = float(np.max(prob) * 100)
    
    # 4. Top spam words contributing to this specific message
    # We find the intersection of the words in the message and model feature importances
    top_spam_words = []
    
    # Attempt to extract weights based on model type
    weights = None
    if hasattr(model, 'feature_log_prob_'):
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0] # log odds for spam
    elif hasattr(model, 'coef_'):
        weights = model.coef_[0]
        if hasattr(weights, 'toarray'):
            weights = weights.toarray()[0]
        else:
            weights = np.asarray(weights).squeeze()
    elif hasattr(model, 'feature_importances_'):
        weights = model.feature_importances_
        
    if weights is not None:
        feature_names = vectorizer.get_feature_names_out()
        # Find features present in this specific message
        message_indices = X_vec[0].nonzero()[1]
        
        # Sort these specific indices by their global model weight
        sorted_indices = sorted(message_indices, key=lambda idx: weights[idx], reverse=True)
        
        # Get top 5 words
        for idx in sorted_indices[:5]:
            top_spam_words.append(feature_names[idx])
            
    return {
        "classification": classification,
        "confidence": confidence,
        "top_spam_words": top_spam_words,
        "cleaned_text": cleaned
    }

if __name__ == "__main__":
    # Test Prediction
    sample = "Congratulations! You've won a $1,000,000 prize! Click here IMMEDIATELY."
    print("Testing sample message...")
    res = predict(sample)
    print(res)
