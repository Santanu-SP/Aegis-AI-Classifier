"""
Text Preprocessing Module for Spam Classifier.
Includes functions for text cleaning and entire dataset preparation.
"""

import string
import re
import pandas as pd
from typing import Tuple, List

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# NLTK resources are expected to be downloaded already.
# nltk.download('stopwords')
# nltk.download('punkt')

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Cleans a given text string for NLP processing:
      - Lowercases
      - Removes punctuation & special characters
      - Removes numbers
      - Tokenizes
      - Removes stopwords
      - Applies stemming
    
    Args:
        text (str): Input text message.
        
    Returns:
        str: The transformed, cleaned text.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2 & 3. Remove punctuation, special characters, and numbers
    # Regex to keep only alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 4. Tokenize
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords
    stop_words = set(stopwords.words('english'))
    # 6. Apply stemming
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    # 7. Return single string
    return " ".join(cleaned_tokens)

def load_and_preprocess_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads raw CSV data and runs full cleaning pipeline.
    
    Args:
        filepath (str): Path to dataset CSV.
    
    Returns:
        pd.DataFrame: Cleaned pandas dataframe with mapped labels.
    """
    if filepath is None:
        import os
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'spam.csv')
    # Load dataset. using 'latin-1' encoding due to typical kaggle SMS dataset configs
    # The file has extra empty columns (Unnamed: 2, etc.), we only need first two
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        df = df.iloc[:, :2]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()
        
    # Renames columns to 'label' and 'message'
    df.columns = ['label', 'message']
    
    # Maps label: ham -> 0, spam -> 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print("Applying text cleaning... This might take a moment.")
    # Applies clean_text() to every message
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    # Ensure there are no null rows created
    df = df.dropna(subset=['label'])
    
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    print(f"Processed Dataset Shape: {df.shape}")
    print(df.head())
