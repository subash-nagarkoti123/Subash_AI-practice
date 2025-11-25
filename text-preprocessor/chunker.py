import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def chunk_text(
    text: str,
    min_tokens=50,
    max_tokens=100,
    overlap=20
):
    """
    Splits text into high-quality chunks while preserving sentences.
    - Each chunk has between min_tokens and max_tokens
    - Overlap tokens are kept from previous chunk
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        token_len = len(tokens)

        # Handle sentences longer than max_tokens
        if token_len > max_tokens:
            split_tokens = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
            for st in split_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = st[-overlap:] if overlap > 0 else []
                current_token_count = len(current_chunk)
            continue

        # If adding this sentence exceeds max_tokens, save current chunk
        if current_token_count + token_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            # Keep overlap tokens for next chunk
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_token_count = len(current_chunk)

        # Add sentence tokens
        current_chunk.extend(tokens)
        current_token_count += token_len

    # Add any remaining tokens as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def save_chunks_to_json(chunks, filename="chunked.json"):
    """
    Saves list of chunks into JSON format:
    [
      {"id": 1, "chunk": "..."},
      {"id": 2, "chunk": "..."},
      ...
    ]
    """
    data = [{"id": idx + 1, "chunk": chunk} for idx, chunk in enumerate(chunks)]
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {filename}")

# ---- Main Program ----
if __name__ == "__main__":
    input_file = "cleaned.txt"
    output_file = "chunked.json"

    # Load JSON and get the cleaned_text
    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        raise ValueError("cleaned.txt is empty! Nothing to chunk.")

    # Generate chunks
    chunks = chunk_text(
        raw_text,
        min_tokens=50,   # minimum tokens per chunk
        max_tokens=100,  # maximum tokens per chunk
        overlap=20       # tokens to carry over for context
    )

    # Save to JSON
    save_chunks_to_json(chunks, output_file)

