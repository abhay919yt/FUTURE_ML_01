"""
train.py
--------
Trains two models on the support ticket dataset:
  1. Category classifier  -> Billing / Technical / Account / Product / Shipping
  2. Priority classifier   -> High / Medium / Low

Pipeline: clean text -> TF-IDF vectorize -> Logistic Regression.

Saves trained models + vectorizers to the models/ folder, and evaluation
reports + confusion matrix images to the outputs/ folder.

Run with:  python src/train.py
"""

import os
import sys
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it works without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_series  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tickets.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["ticket_text", "category", "priority"])
    return df


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix -> {out_path}")


def train_classifier(df, target_col, label_prefix):
    print(f"\n{'=' * 60}\nTraining {label_prefix} classifier (target: {target_col})\n{'=' * 60}")

    X_text = clean_series(df["ticket_text"].tolist())
    y = df[target_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}\n")
    report = classification_report(y_test, y_pred)
    print(report)

    labels = sorted(set(y))
    plot_confusion_matrix(y_test, y_pred, labels, f"{label_prefix} Confusion Matrix",
                           f"{label_prefix.lower()}_confusion_matrix.png")

    report_path = os.path.join(OUTPUTS_DIR, f"{label_prefix.lower()}_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{label_prefix} Classifier Evaluation\n")
        f.write(f"Accuracy: {acc:.2%}\n\n")
        f.write(report)
    print(f"Saved evaluation report -> {report_path}")

    joblib.dump(model, os.path.join(MODELS_DIR, f"{label_prefix.lower()}_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, f"{label_prefix.lower()}_vectorizer.pkl"))
    print(f"Saved model + vectorizer -> models/{label_prefix.lower()}_model.pkl")

    return model, vectorizer, acc


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} tickets.")
    print("\nCategory distribution:\n", df["category"].value_counts())
    print("\nPriority distribution:\n", df["priority"].value_counts())

    cat_model, cat_vec, cat_acc = train_classifier(df, "category", "Category")
    pri_model, pri_vec, pri_acc = train_classifier(df, "priority", "Priority")

    print(f"\n{'=' * 60}")
    print("DONE. Summary:")
    print(f"  Category classifier accuracy : {cat_acc:.2%}")
    print(f"  Priority classifier accuracy : {pri_acc:.2%}")
    print(f"  Models saved in   : {MODELS_DIR}")
    print(f"  Reports saved in  : {OUTPUTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
