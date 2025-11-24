from transformers import pipeline

check = pipeline("text-generation", model="gpt2")

res = check(
        "I am looking for",
        max_length=20,
        num_return_sequences=3,
        )
print(res)


