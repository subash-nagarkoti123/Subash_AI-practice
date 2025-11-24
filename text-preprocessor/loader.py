# loader.py
# -----------------------------------------
# Purpose: Load raw text from a file.
# Usage: python3 loader.py
# -----------------------------------------

def load_text(file_path: str) -> str:
    """
    Reads raw text from a local file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# Standalone Run Test
if __name__ == "__main__":
    path = "../input/messy_1.txt"  # Modify as needed
    text = load_text(path)
    print("\n--- Loaded Text ---\n")
    print(text[:500])  # Print first 500 chars only

