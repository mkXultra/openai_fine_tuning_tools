from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load LaBSE model and tokenizer
model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example usage
ja_text = "こんにちは、世界"
en_text = "Hello, world"

ja_embedding = get_embedding(ja_text)
en_embedding = get_embedding(en_text)

similarity = cosine_similarity(ja_embedding, en_embedding)
print(f"Similarity between '{ja_text}' and '{en_text}': {similarity}")

# Compare with different texts
ja_text2 = "今日は良い天気です"
en_text2 = "The weather is nice today"

ja_embedding2 = get_embedding(ja_text2)
en_embedding2 = get_embedding(en_text2)

similarity2 = cosine_similarity(ja_embedding2, en_embedding2)
print(f"Similarity between '{ja_text2}' and '{en_text2}': {similarity2}")