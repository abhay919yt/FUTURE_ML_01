"""
classify_ticket.py
-------------------
Loads the trained models and classifies new support ticket text into:
  - a category (Billing, Technical, Account, Product, Shipping)
  - a priority level (High, Medium, Low)

Two ways to use it:

1) Interactive mode (just run it, then type tickets one at a time):
       python src/classify_ticket.py

2) One-off classification from the command line:
       python src/classify_ticket.py "My card was charged twice, refund me now!"

Priority logic note:
  The priority *model* is trained on the language patterns in the ticket text.
  On top of that, we apply a small rule-based safety net: if the ticket contains
  clear urgency keywords (e.g. "urgent", "immediately", "down", "not working"),
  the priority is nudged up. This mirrors how real support teams combine
  ML predictions with simple business rules, and keeps critical tickets from
  ever slipping through as "Low".
"""

import os
import re
import sys
import joblib

sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_text  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

URGENCY_KEYWORDS = [
    "urgent", "urgently", "immediately", "asap", "emergency", "critical",
    "right now", "down", "not working", "cannot access", "can't access",
    "hacked", "unauthorized", "overdue", "deadline", "broken", "crash",
]

PRIORITY_RANK = {"Low": 0, "Medium": 1, "High": 2}


def _load_pickle(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            f"Have you run 'python src/train.py' yet? That step creates the model files."
        )
    return joblib.load(path)


def load_models():
    return {
        "category_model": _load_pickle("category_model.pkl"),
        "category_vectorizer": _load_pickle("category_vectorizer.pkl"),
        "priority_model": _load_pickle("priority_model.pkl"),
        "priority_vectorizer": _load_pickle("priority_vectorizer.pkl"),
    }


def apply_urgency_boost(text: str, predicted_priority: str) -> str:
    lowered = text.lower()
    hits = [kw for kw in URGENCY_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", lowered)]
    if hits and PRIORITY_RANK[predicted_priority] < PRIORITY_RANK["High"]:
        return "High"
    return predicted_priority


def classify(text: str, models: dict) -> dict:
    cleaned = clean_text(text)

    cat_vec = models["category_vectorizer"].transform([cleaned])
    category = models["category_model"].predict(cat_vec)[0]
    cat_confidence = models["category_model"].predict_proba(cat_vec).max()

    pri_vec = models["priority_vectorizer"].transform([cleaned])
    priority_raw = models["priority_model"].predict(pri_vec)[0]
    pri_confidence = models["priority_model"].predict_proba(pri_vec).max()

    priority_final = apply_urgency_boost(text, priority_raw)

    return {
        "ticket_text": text,
        "category": category,
        "category_confidence": round(float(cat_confidence), 3),
        "priority_model": priority_raw,
        "priority_final": priority_final,
        "priority_confidence": round(float(pri_confidence), 3),
        "urgency_rule_applied": priority_final != priority_raw,
    }


def print_result(result: dict):
    print("\n--- Classification Result ---")
    print(f"Ticket    : {result['ticket_text']}")
    print(f"Category  : {result['category']}  (confidence: {result['category_confidence']:.0%})")
    if result["urgency_rule_applied"]:
        print(f"Priority  : {result['priority_final']}  "
              f"(model said {result['priority_model']}, bumped up due to urgency keywords)")
    else:
        print(f"Priority  : {result['priority_final']}  (confidence: {result['priority_confidence']:.0%})")
    print("-" * 30)


def main():
    models = load_models()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = classify(text, models)
        print_result(result)
        return

    print("Support Ticket Classifier - interactive mode")
    print("Type a ticket message and press Enter. Type 'quit' to exit.\n")
    while True:
        text = input("Ticket text > ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not text:
            continue
        result = classify(text, models)
        print_result(result)


if __name__ == "__main__":
    main()
