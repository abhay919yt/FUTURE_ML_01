# Customer Support Ticket Classification System

Automatically classifies customer support tickets by **category** (Billing, Technical,
Account, Product, Shipping) and **priority** (High, Medium, Low), so support teams can
respond faster and route tickets to the right place.

---

## What's inside

```
ticket_classification_system/
├── data/
│   ├── tickets.csv              <- sample training data (150 tickets)
│   └── generate_dataset.py      <- script that created tickets.csv (optional to re-run)
├── src/
│   ├── preprocessing.py         <- text cleaning & tokenization
│   ├── train.py                 <- trains both models, evaluates them
│   └── classify_ticket.py       <- classify new tickets (command line tool)
├── notebook/
│   └── Ticket_Classification_System.ipynb   <- full walkthrough with charts
├── models/                      <- trained models get saved here (created automatically)
├── outputs/                     <- evaluation reports & charts get saved here (created automatically)
├── run_me.py                    <- ONE COMMAND that does everything
├── requirements.txt
└── README.md                    <- you are here
```

---

## Super simple steps (recommended)

**Step 1 — Unzip this folder** anywhere on your computer.

**Step 2 — Open a terminal / command prompt** in that folder.
(Windows: open the folder, click the address bar, type `cmd`, press Enter.
Mac: right-click the folder → "New Terminal at Folder", or open Terminal and `cd` into it.)

**Step 3 — Install the requirements** (only needed once):
```
pip install -r requirements.txt
```

**Step 4 — Run the whole thing with one command:**
```
python run_me.py
```

That's it. This will:
- Train the category and priority models
- Print accuracy and evaluation results
- Save the trained models into `models/`
- Save charts and reports into `outputs/`
- Show you a live demo classifying a few example tickets

---

## Classify your own tickets afterward

Once you've run `run_me.py` at least once (so the models exist), classify any ticket text:

```
python src/classify_ticket.py "My internet has been down for 2 hours, please help ASAP!"
```

Or run it with no text to enter tickets one at a time:
```
python src/classify_ticket.py
```

---

## Want the full notebook walkthrough with charts?

```
jupyter notebook notebook/Ticket_Classification_System.ipynb
```
Then in the browser tab that opens, click **Cell → Run All** (or press Shift+Enter on each
cell from top to bottom). This shows the same system step by step, with data exploration
charts and confusion matrices included inline.

---

## Using your own ticket data

Replace `data/tickets.csv` with your own file. It just needs these three columns:

| ticket_text | category | priority |
|---|---|---|
| "My card was charged twice..." | Billing | High |

Then re-run `python run_me.py` (or the notebook) to retrain on your real data.

---

## How priority is decided

Priority combines two things:
1. **The trained ML model**, which learns urgency patterns from the ticket wording.
2. **A simple keyword safety net** — if a ticket contains obvious urgency words
   ("urgent", "immediately", "system is down", "not working", "hacked", etc.), the
   priority is automatically bumped up to High. This mirrors how real support teams
   combine machine learning with straightforward business rules, so critical issues
   never slip through.

## Notes

- If `pip install -r requirements.txt` fails on `nltk` data downloads, don't worry —
  `preprocessing.py` automatically falls back to a built-in text cleaner, so the project
  still works fully offline.
- All 4 model files (`models/*.pkl`) and the evaluation charts/reports (`outputs/*`) are
  regenerated every time you run `run_me.py` or `src/train.py`.
