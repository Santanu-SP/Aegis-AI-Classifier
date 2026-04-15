"""
Training Module for Spam Classifier.
Handles TF-IDF Vectorization, multi-model training/evaluation, and plotting.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from preprocess import load_and_preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def train_and_evaluate(df):
    """
    Executes Feature extraction, Model training, Evaluation, Plotting, and saving best model.
    """
    X = df['cleaned_message']
    y = df['label'].values

    print("Step 1: Feature Extraction using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_vec = vectorizer.fit_transform(X)

    # Save Vectorizer
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    print("Saved vectorizer to models/tfidf_vectorizer.pkl")

    print("Step 2: Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (Linear Kernel)": SVC(kernel='linear', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    best_f1 = 0
    best_model_name = ""
    best_model = None
    
    # Setup for ROC Curve Plot
    plt.figure(figsize=(10, 8))
    
    print("\nStep 3: Training Models...")
    for name, model in models.items():
        print(f"Training [{name}]...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append([name, acc, prec, rec, f1, auc])

        # Plot ROC curve for this model
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

        # Track best model based on F1 Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # Finalize ROC Curve Plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'))
    plt.close()

    print("\n=== All Models Comparison ===")
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    print(tabulate(results, headers=headers, tablefmt="github"))

    print(f"\nSaving best model ({best_model_name}) with F1 ({best_f1:.4f})...")
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'spam_classifier.pkl'))

    # Step 4: Generate additional Evaluation Plots
    generate_plots(df, y_test, X_test, best_model, vectorizer, results, headers)


def generate_plots(df, y_test, X_test, best_model, vectorizer, results, headers):
    print("Step 4: Generating plots...")

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='Set2')
    plt.xticks(ticks=[0, 1], labels=['Ham (0)', 'Spam (1)'])
    plt.title('Class Distribution (Ham vs Spam)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'))
    plt.close()

    # 2. Confusion Matrix (Best Model)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix (Best Model)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    plt.close()

    # 3. Model Comparison Bar Chart
    # Convert results to dataframe for easy plotting
    df_results = pd.DataFrame(results, columns=headers)
    df_melt = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")
    # Filter out ROC-AUC just to keep Acc/Prec/Rec/F1 for grouped bars
    df_melt = df_melt[df_melt["Metric"] != "ROC-AUC"]
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melt, palette="viridis")
    plt.title("Model Performance Comparison")
    plt.ylim(0.7, 1.05)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'))
    plt.close()

    # 4. Top 20 Most Important Features (If Naive Bayes)
    importances = None
    if isinstance(best_model, MultinomialNB):
        importances = best_model.feature_log_prob_[1, :]
    elif hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = best_model.coef_[0]
        if hasattr(importances, 'toarray'):
            importances = importances.toarray()[0]
        else:
            importances = np.asarray(importances).squeeze()

    if importances is not None:
        feature_names = vectorizer.get_feature_names_out()
        # Get top 20 indices
        top_indices = np.argsort(importances)[-20:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = [importances[i] for i in top_indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(20), top_importances, color='salmon')
        plt.yticks(range(20), top_features)
        plt.xlabel('Importance / Weight')
        plt.title('Top 20 Spam Indicator Words')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'))
        plt.close()
    else:
        print("Couldn't extract feature importances from best model.")

if __name__ == "__main__":
    df = load_and_preprocess_data()
    if not df.empty:
        train_and_evaluate(df)
        print("Training complete and generated plots in plots/")
    else:
        print("Dataset failed to load. Please ensure data/spam.csv is present.")
