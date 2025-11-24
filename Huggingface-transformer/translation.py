from transformers import pipeline

check = pipeline("translation_en_to_fr")

sentences = ["i am very happy","i am from nepal"]

res = check(sentences, max_length=20)

for t in res:
    print(t['translation_text'])

