import re
import nltk
from nltk.corpus import stopwords

# Download stopwords on first run
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w not in STOP]
    return " ".join(words)
