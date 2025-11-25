import json
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords if not already downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')

def clean_text(text: str) -> str:
    """
    Cleans messy real-world text by:
    - Removing URLs
    - Lowercasing
    - Removing non-alphanumeric characters
    - Removing stopwords
    - Normalizing whitespace
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Lowercase
    text = text.lower()

    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize words
    tokens = word_tokenize(text)

    # Remove stopwords
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in sw]

    # Join back to text
    return " ".join(tokens)


if __name__ == "__main__":
    # Read messy.txt
    with open("input_folder/messy.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Process text
    cleaned = clean_text(raw_text)

    # Save cleaned output to cleaned.json
    with open("cleaned.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)
    
    print("✅ Cleaning complete! Saved to cleaned.txt")

