import os
import json
from tqdm import tqdm
from cleaner import clean_text
from chunker import chunk_text
from sentence_transformers import SentenceTransformer

def load_text_file(path):
    """Load a plain text file, ignoring errors."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def generate_embeddings(chunks):
    """Creates vector embeddings for a list of chunk texts."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)


def preprocess_folder(input_folder, output_file, min_tokens=50, max_tokens=100, overlap=10):
    """
    Full pipeline:
        1. Load files
        2. Clean text
        3. Chunk text
        4. Generate embeddings
        5. Save JSONL
    """
    final_records = []

    # Step 1: Process each text file
    for file in tqdm(os.listdir(input_folder)):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(input_folder, file)
        raw_text = load_text_file(path)

        # Step 2: Clean text
        cleaned = clean_text(raw_text)

        # Step 3: Chunk text
        chunks = chunk_text(
            cleaned,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap=overlap
        )

        # Step 4: Create preliminary records (ID + text + metadata)
        for i, chunk in enumerate(chunks):
            final_records.append({
                "id": f"{file}_chunk_{i}",
                "text": chunk,
                "metadata": {"source": file}
            })

    print(f"Total chunks to embed: {len(final_records)}")

    # Step 5: Generate embeddings
    chunk_texts = [rec["text"] for rec in final_records]
    chunk_ids = [rec["id"] for rec in final_records]
    metadata_list = [rec.get("metadata", {}) for rec in final_records]

    embeddings = generate_embeddings(chunk_texts)

    # Step 6: Save embedding-ready JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for cid, text, meta, emb in zip(chunk_ids, chunk_texts, metadata_list, embeddings):
            record = {
                "id": cid,
                "text": text,
                "metadata": meta,
                "embedding": emb.tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(final_records)} embedding-ready chunks to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess folder of text files to embedding-ready JSONL")
    parser.add_argument("input_folder", type=str, help="Folder containing raw .txt files")
    parser.add_argument("output_file", type=str, help="Output JSONL file path")
    parser.add_argument("--min_tokens", type=int, default=50, help="Minimum tokens per chunk")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=10, help="Number of overlapping tokens per chunk")
    args = parser.parse_args()

    preprocess_folder(args.input_folder, args.output_file, args.min_tokens, args.max_tokens, args.overlap)

