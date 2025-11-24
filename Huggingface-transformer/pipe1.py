from transformers import pipeline

check = pipeline("sentiment-analysis")

res = check(" I am happy working with Huggingface transformer")

print(res)
