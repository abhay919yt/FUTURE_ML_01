"""
preprocessing.py
-----------------
Text cleaning & tokenization for support tickets.

We try to use NLTK for tokenization, stopword removal, and lemmatization
(as required by the project spec). NLTK needs small data packages
('punkt', 'stopwords', 'wordnet') downloaded once -- this file downloads
them automatically the first time it runs.

If, for any reason, NLTK's data can't be downloaded (no internet, blocked
network, etc.), this file automatically falls back to an equivalent
built-in cleaner so the whole project still runs without crashing.
"""

import re
import string

# ---------------------------------------------------------------
# Try to set up NLTK. Fall back gracefully if unavailable.
# ---------------------------------------------------------------
_NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    def _ensure_nltk_data():
        packages = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
        ]
        for path, pkg in packages:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

    _ensure_nltk_data()
    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_READY = True
except Exception:
    _NLTK_READY = False

# ---------------------------------------------------------------
# Built-in fallback (used automatically if NLTK isn't available)
# ---------------------------------------------------------------
_FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're",
    "you've", "your", "yours", "yourself", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their", "what", "which", "who", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "please",
    "hi", "hello", "hey", "thanks", "thank",
}


def clean_text(text: str) -> str:
    """
    Cleans and tokenizes a single piece of text.
    Steps: lowercase -> remove punctuation/numbers -> tokenize ->
           remove stopwords -> lemmatize (if NLTK available) -> rejoin.

    Returns a cleaned string ready for vectorization (e.g. TF-IDF).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep letters only
    text = re.sub(r"\s+", " ", text).strip()

    if _NLTK_READY:
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    else:
        tokens = text.split()
        tokens = [t.strip(string.punctuation) for t in tokens]
        tokens = [t for t in tokens if t and t not in _FALLBACK_STOPWORDS and len(t) > 1]

    return " ".join(tokens)


def clean_series(texts):
    """Applies clean_text() to a list/Series of texts. Returns a list."""
    return [clean_text(t) for t in texts]


if __name__ == "__main__":
    sample = "Hi team, I was charged TWICE for my subscription this month!! Please refund me ASAP, this is urgent."
    print("Using NLTK:", _NLTK_READY)
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
