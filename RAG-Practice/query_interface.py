import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# Configuration
# -------------------------------
EMBEDDINGS_FILE = "output.jsonl"  # Precomputed embeddings
INDEX_FILE = "faiss.index"        # Pre-built FAISS index
TOP_K = 3                         # Number of top chunks to retrieve
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # For query embedding

LOCAL_GEN_MODEL = "microsoft/phi-2"
DEVICE = "cpu"                     # force CPU
MAX_GEN_TOKENS = 200               # reduce token length for CPU speed

# -------------------------------
# Load chunk texts and sources
# -------------------------------
def load_texts_and_sources(jsonl_file):
    texts, sources = [], []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            sources.append(record["metadata"].get("source", "unknown"))
    return texts, sources

# -------------------------------
# Retrieval function
# -------------------------------
def retrieve(query, model, index, texts, sources, top_k=TOP_K):
    """
    Retrieve top-k relevant chunks with source info.
    """
    query_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    results = [(texts[i], sources[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# -------------------------------
# Local LLM generation
# -------------------------------
def generate_answer_local(context, query, tokenizer, model):
    """
    Generate answer from local LLM using retrieved context.
    """
    prompt = f"Answer the question using the following context and cite sources:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_GEN_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    answer_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt from output
    if answer_text.startswith(prompt):
        answer_text = answer_text[len(prompt):]
    return answer_text.strip()

# -------------------------------
# Main interactive interface
# -------------------------------
def main():
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)
    print(f"✅ Loaded FAISS index from {INDEX_FILE}")

    print("Loading chunk texts and sources...")
    texts, sources = load_texts_and_sources(EMBEDDINGS_FILE)

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Loading local generation model '{LOCAL_GEN_MODEL}' on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_GEN_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(LOCAL_GEN_MODEL, device_map={"": DEVICE})
    gen_model.eval()
    print("✅ Phi-2 model loaded successfully!\n")

    print("✅ Query interface ready! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Retrieve top-k chunks
        results = retrieve(query, emb_model, index, texts, sources, TOP_K)

        # Prepare context and cited sources
        context_texts = [f"[{i+1}] {txt}" for i, (txt, _, _) in enumerate(results)]
        cited_sources = [f"[{i+1}] {src}" for i, (_, src, _) in enumerate(results)]
        context = "\n".join(context_texts)

        # Generate answer locally
        answer = generate_answer_local(context, query, tokenizer, gen_model)

        # Display output
        print("\n--- Generated Answer ---")
        print(answer)
        print("\n--- Cited Sources ---")
        print("\n".join(cited_sources))
        print("------------------------")

if __name__ == "__main__":
    main()
