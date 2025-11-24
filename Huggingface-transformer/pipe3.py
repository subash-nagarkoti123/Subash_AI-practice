from transformers import pipeline

check = pipeline("zero-shot-classification")

res = check(
        "Lionel Messi is the Greatest Player of All time",
        candidate_labels=["Football", "Cricket", "Volleyball"],
        )
print(res)

