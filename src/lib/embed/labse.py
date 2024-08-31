from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class LaBSEEmbedder:
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def compare_texts(self, text1, text2):
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        similarity = self.cosine_similarity(embedding1, embedding2)
        return similarity

# 使用例
if __name__ == "__main__":
    embedder = LaBSEEmbedder()

    ja_text = "こんにちは、世界"
    en_text = "Hello, world"
    similarity = embedder.compare_texts(ja_text, en_text)
    print(f"Similarity between '{ja_text}' and '{en_text}': {similarity}")

    ja_text2 = "今日は良い天気です"
    en_text2 = "The weather is nice today"
    similarity2 = embedder.compare_texts(ja_text2, en_text2)
    print(f"Similarity between '{ja_text2}' and '{en_text2}': {similarity2}")