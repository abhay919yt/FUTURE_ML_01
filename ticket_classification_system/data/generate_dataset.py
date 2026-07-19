"""
generate_dataset.py
--------------------
Creates a synthetic but realistic customer support ticket dataset
and saves it as data/tickets.csv.

Columns:
    ticket_text : the raw text of the customer's support message
    category    : the type of issue (Billing, Technical, Account, Product, Shipping)
    priority    : how urgent the ticket is (High, Medium, Low)

You normally do NOT need to run this yourself -- data/tickets.csv is
already included in this project. It's here so you can see exactly
how the data was created, or regenerate/expand it if you want more rows.
"""

import random
import csv
import os

random.seed(42)

# ---------------------------------------------------------------
# Templates: category -> list of (text_template, priority)
# We mix urgent/angry language (High), normal issues (Medium),
# and simple questions / minor requests (Low).
# ---------------------------------------------------------------

TICKETS = {
    "Billing": {
        "High": [
            "I was charged twice for my subscription this month and need a refund immediately, this is unacceptable.",
            "My credit card was billed for {amount} but I cancelled last week, please fix this urgently, I need my money back now.",
            "I have been charged an incorrect amount of {amount} and my account is now overdrawn because of it, please resolve today.",
            "This is the third time you have overcharged me, I want a full refund right now or I am disputing the charge with my bank.",
            "Emergency: an unauthorized charge of {amount} just appeared on my card, please investigate and refund immediately.",
        ],
        "Medium": [
            "I noticed a charge of {amount} on my statement that I don't recognize, can you explain what it's for?",
            "Can someone check why my invoice this month is higher than usual? It shows {amount} instead of the normal amount.",
            "I would like to update my billing address and payment method on file.",
            "My last payment of {amount} didn't seem to go through, can you confirm if it was received?",
            "I want to switch from monthly to annual billing, can you help me with that?",
        ],
        "Low": [
            "Could you send me a copy of last month's invoice for my records?",
            "Just wondering when my next billing date is.",
            "How do I download my past receipts from the billing portal?",
            "Is there a discount available if I refer a friend?",
            "What payment methods do you currently accept?",
        ],
    },
    "Technical": {
        "High": [
            "The app keeps crashing every time I open it and I cannot access any of my data, this is blocking my work completely.",
            "Production is down, we are getting a 500 error on every request since this morning, please help ASAP.",
            "I cannot log in at all, it just says 'server error' and my whole team is stuck, this is extremely urgent.",
            "The system deleted all my saved files after the last update, I need this fixed today, I'm losing work.",
            "Critical bug: the checkout page throws an error and no customers can complete a purchase right now.",
        ],
        "Medium": [
            "The search feature is returning irrelevant results, can someone look into this?",
            "I'm getting an error message when I try to export my report to PDF.",
            "The dashboard is loading very slowly today, is there a known issue?",
            "After the recent update, the notification settings reset to default, can you help me fix this?",
            "I'm having trouble syncing my data between the mobile app and the desktop version.",
        ],
        "Low": [
            "Just a small suggestion, it would be nice if the font size could be adjusted in settings.",
            "How do I enable dark mode in the app?",
            "Is there a keyboard shortcut to quickly search within the app?",
            "The icon in the top right looks slightly misaligned on my screen, not urgent though.",
            "Can you point me to the documentation for the API?",
        ],
    },
    "Account": {
        "High": [
            "My account has been locked and I cannot access anything, I need this resolved immediately, I have an important deadline.",
            "I think my account has been hacked, there are login attempts I don't recognize, please secure it right now.",
            "I was suddenly logged out of all devices and cannot get back in, this is urgent as I need it for a live event today.",
            "Someone changed my account email without my permission, please restore access immediately, this is a security issue.",
            "My account was suspended without explanation and I need it reinstated urgently, I rely on it daily.",
        ],
        "Medium": [
            "I forgot my password and the reset link isn't arriving in my inbox, can you help?",
            "I would like to change the email address associated with my account.",
            "Can you help me merge two accounts I accidentally created?",
            "I want to update my profile information but the save button isn't working.",
            "I need to add a second user to my account, how do I do that?",
        ],
        "Low": [
            "How do I change my display name on my profile?",
            "Just checking, can I have multiple accounts under the same email?",
            "What's the process to permanently delete my account if I ever wanted to?",
            "Can you tell me how to enable two factor authentication?",
            "Is it possible to change my username later on?",
        ],
    },
    "Product": {
        "High": [
            "The product I received is completely broken and unusable, I need a replacement urgently before my event this weekend.",
            "This is a safety issue, the device overheated and started smoking, please advise immediately.",
            "I received the wrong item entirely and I need the correct one urgently, this was for a time sensitive gift.",
            "The product stopped working within a day of use and I have an important deadline relying on it, please help now.",
            "Major defect found, the item arrived damaged and leaking, I need an urgent replacement or refund.",
        ],
        "Medium": [
            "The product doesn't quite match the description on the website, can you clarify the specifications?",
            "I'd like to know if this product is compatible with older models before I purchase an upgrade.",
            "The instructions manual seems to be missing a few pages, can you send me a digital copy?",
            "Can you tell me more about the warranty coverage for this product?",
            "I want to exchange this item for a different size, what's the process?",
        ],
        "Low": [
            "Do you have this product in other colors?",
            "Just curious what materials this product is made from.",
            "Can you recommend accessories that go well with this product?",
            "Is there a user community or forum for tips on using this product?",
            "What's the average lifespan of this product with normal use?",
        ],
    },
    "Shipping": {
        "High": [
            "My package was marked as delivered but I never received it and I need it urgently for tomorrow's event.",
            "The delivery is a week late and I have already reached out twice with no response, this needs urgent attention.",
            "My order shows as lost in transit and I need it replaced immediately, this was time sensitive.",
            "The courier left my package outside in the rain and it's completely damaged, I need a resolution right away.",
            "I paid extra for express shipping but it still hasn't arrived and it's already 3 days late, this is urgent.",
        ],
        "Medium": [
            "Can you provide an updated tracking number, the one I have doesn't seem to be working?",
            "I'd like to change my shipping address before the order ships out.",
            "My order is delayed, can you give me an estimated new delivery date?",
            "Can I combine two separate orders into a single shipment to save on shipping costs?",
            "The tracking hasn't updated in three days, can you check on the status?",
        ],
        "Low": [
            "How long does standard shipping usually take for international orders?",
            "Do you ship to PO boxes?",
            "Just wondering what shipping carriers you typically use.",
            "Can I schedule a specific delivery time for my package?",
            "What are the shipping costs for orders over a certain amount?",
        ],
    },
}

AMOUNTS = ["$49.99", "$120.00", "$15.50", "$89.00", "$250.00", "$12.99", "$300.00", "$45.00"]

GREETINGS = ["", "Hi team, ", "Hello, ", "Hi, ", "To whom it may concern, ", "Hey, ", ""]
SIGNOFFS = ["", " Thanks.", " Please advise.", " Looking forward to your response.", " Thank you.", ""]


def build_dataset(rows_per_combo=8):
    records = []
    for category, priorities in TICKETS.items():
        for priority, templates in priorities.items():
            for _ in range(rows_per_combo):
                template = random.choice(templates)
                text = template.format(amount=random.choice(AMOUNTS))
                text = random.choice(GREETINGS) + text + random.choice(SIGNOFFS)
                records.append({"ticket_text": text, "category": category, "priority": priority})
    random.shuffle(records)
    return records


def main():
    records = build_dataset(rows_per_combo=10)
    out_path = os.path.join(os.path.dirname(__file__), "tickets.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_text", "category", "priority"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Generated {len(records)} tickets -> {out_path}")


if __name__ == "__main__":
    main()
