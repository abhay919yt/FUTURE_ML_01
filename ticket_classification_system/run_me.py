"""
run_me.py
---------
The single "just run this" entry point for the whole project.

It will:
  1. Train both models (category + priority) on data/tickets.csv
  2. Save the trained models into models/
  3. Save evaluation reports + confusion matrix charts into outputs/
  4. Run a live demo, classifying a few example tickets so you can see it work

Usage:
    python run_me.py
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60)
    print("STEP 1/2: Training the models (this takes a few seconds)")
    print("=" * 60)
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "src", "train.py")], check=True)

    print("\n" + "=" * 60)
    print("STEP 2/2: Demo - classifying a few example tickets")
    print("=" * 60)
    demo_tickets = [
        "I was charged twice this month and need a refund immediately, this is unacceptable!",
        "Can you tell me what colors this product comes in?",
        "The website has been down for an hour and customers cannot check out, please help ASAP.",
    ]
    from src.classify_ticket import load_models, classify, print_result
    models = load_models()
    for t in demo_tickets:
        result = classify(t, models)
        print_result(result)

    print("\nAll done! Your trained models are saved in the 'models/' folder.")
    print("Evaluation reports and charts are saved in the 'outputs/' folder.")
    print("\nTo classify your own tickets, run:")
    print('    python src/classify_ticket.py "your ticket text here"')
    print("or for interactive mode:")
    print("    python src/classify_ticket.py")
    print("\nTo see the full walkthrough with charts, open:")
    print("    notebook/Ticket_Classification_System.ipynb")


if __name__ == "__main__":
    main()
