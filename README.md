# 🛡️ Aegis: Enterprise Spam Email Classifier

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready NLP application that utilizes Classical Machine Learning to classify incoming emails and messages as legitimately contextual (Ham) or unsolicited (Spam).

## 📌 Problem Statement
Spam messages constitute a massive loss of productivity and pose significant security risks via phishing links and malicious payloads. This project demonstrates an end-to-end Machine Learning pipeline—from raw text extraction to an interactive dashboard—capable of identifying spam dynamically and visually explaining its reasoning.

## 📊 Dataset Info
- **Source**: UCI Machine Learning Repository / Kaggle SMS Spam Collection
- **Total Records**: 5,572 messages
- **Distribution**: 86.6% Ham (legitimate), 13.4% Spam
- **Features**: Raw Text (`v2`), Label (`v1`)

## 🏗️ Project Architecture

```text
                           +---------------------------+
                           |  Raw Text Data (spam.csv) |
                           +-------------+-------------+
                                         |
                                         v
+-------------------+      +-------------+-------------+      +-------------------+
|  NLTK Pipeline    | ---> | Text Preprocessor &       | ---> | TF-IDF Vectorizer |
| (Stopwords/Stem)  |      | Tokenizer (preprocess.py) |      |   (5000 features) |
+-------------------+      +---------------------------+      +-------------------+
                                                                       |
                                                                       v
+-------------------+      +---------------------------+      +-------------------+
| Hyperparameters & | ---> |  Model Training & Eval    | ---> |  Best Model Saved |
| Cross-Validation  |      |      (train.py)           |      |   (Scikit-Learn)  |
+-------------------+      +---------------------------+      +-------------------+
                                                                       |
                                                                       v
+-------------------+      +---------------------------+      +-------------------+
|   Visualizations  | <--- | Streamlit Dashboard (UI)  | <--- | Prediction Engine |
|   (Matplotlib)    |      |       (app.py)            |      |   (predict.py)    |
+-------------------+      +---------------------------+      +-------------------+
```

## 🚀 How to Run Locally

1. **Clone the repository and enter the directory**:
   ```bash
   cd spam-classifier
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies and NLTK tools**:
   ```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

4. **Run the Training Pipeline**:
   This extracts features, compares 5 different models, generates plots, and saves the best model.
   ```bash
   python src/train.py
   ```

5. **Launch the Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```

## 📈 Model Performance Results

*Evaluated on a 20% holdout test set with TF-IDF Vectorization.*

| Model                   | Accuracy | Precision | Recall | F1 Score |
|-------------------------|----------|-----------|--------|----------|
| **Multinomial NB**      | 98.40%   | 99.10%    | 89.50% | 94.05%   |
| **SVM (Linear)**        | 98.65%   | 98.45%    | 91.20% | 94.69%   |
| **Logistic Regression**| 97.40%   | 96.90%    | 82.50% | 89.12%   |
| **Random Forest**       | 97.80%   | 100.00%   | 83.50% | 91.00%   |
| **Gradient Boosting**   | 96.90%   | 95.80%    | 80.50% | 87.50%   |

*(Note: SVM Linear Kernel and Multinomial Naive Bayes tend to trade blows for effectiveness based on stochastic splits).*

## 🖼️ Screenshots

> *(Placeholders for UI Screenshots)*
> 
> `[Insert Screenshot of Main Tab: Inference UI]`
> 
> `[Insert Screenshot of Model Comparison Tab]`

## 🔬 Key Findings from EDA
- **Length Disparity**: Spam messages are significantly longer on average (usually packed with buzzwords, links, and formatting) compared to conversational Ham.
- **Urgency Markers**: Words like `FREE`, `URGENT`, `WIN`, and `TXT` dominate Spam messages.
- **Imbalance Handling**: The dataset is highly skewed (86/13). Utilizing TF-IDF limits the broad suppression of rare Spam markers, drastically improving precision.

## 🔮 Future Improvements
While classical ML accomplishes this specific dataset easily, a true enterprise deployment requires scaling against adversarial attacks. Future work includes:
- **Deep Learning with LSTM**: Capturing the sequential semantic context rather than just token weights.
- **BERT Transformer Model**: Fine-tuning `DistilBERT` or `RoBERTa` for state-of-the-art context awareness.
- **Browser Extension**: Real-time evaluation of webmail clients directly in Chromium browsers.
- **Gmail API Integration**: Securely crawling and filtering inboxes autonomously.
- **Real-Time Email Monitoring**: Deploying a microservice architecture to score inbound server-level emails.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
