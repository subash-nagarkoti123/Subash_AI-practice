import json
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks):
    """
    Creates vector embeddings for each text chunk.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)


def main(input_file="chunked.json", output_file="formatted_chunks.jsonl"):
    # Load chunks from JSON
    with open(input_file, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    # Extract chunk texts, ids, and metadata
    chunk_texts = [item.get("chunk") or item.get("text") for item in chunk_data]
    chunk_ids = [item["id"] for item in chunk_data]
    metadata_list = [item.get("metadata", {}) for item in chunk_data]

    # Generate embeddings
    embeddings = generate_embeddings(chunk_texts)

    # Prepare embedding-ready JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for cid, text, meta, emb in zip(chunk_ids, chunk_texts, metadata_list, embeddings):
            record = {
                "id": cid,
                "text": text,
                "metadata": meta,
                "embedding": emb.tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(chunk_texts)} embedding-ready records to {output_file}")


if __name__ == "__main__":
    main()

