import os
from openai import OpenAI
import numpy as np

# OpenAI APIキーの設定
# os.environ["OPENAI_API_KEY"] = "あなたのAPIキー"

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar(target, candidates):
    target_embedding = get_embedding(target)
    candidate_embeddings = [get_embedding(candidate) for candidate in candidates]
    
    similarities = [cosine_similarity(target_embedding, cand_emb) for cand_emb in candidate_embeddings]
    
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[most_similar_index]

# サンプルテキスト
target = "東京は日本の首都で、世界有数の大都市です。"
candidates = [
    "東京は日本の中心地であり、多くの人々が住んでいます。",
    "京都は日本の古都として知られ、多くの観光客が訪れます。",
    "大阪は関西地方最大の都市で、商業の中心地です。",
    "東京は政治と経済の中心地であり、日本最大の都市圏を形成しています。",
    "富士山は日本の象徴的な山で、多くの人々に愛されています。"
]

# 最も類似した候補を見つける
most_similar_index, similarity = find_most_similar(target, candidates)

# 結果の表示
print(f"ターゲット: {target}\n")
print("候補テキスト:")
for i, candidate in enumerate(candidates):
    print(f"{i+1}. {candidate}")

print(f"\n最も類似した候補:")
print(f"候補 {most_similar_index + 1}: {candidates[most_similar_index]}")
print(f"類似度: {similarity:.4f}")

# すべての候補との類似度を表示
print("\nすべての候補との類似度:")
for i, candidate in enumerate(candidates):
    _, sim = find_most_similar(target, [candidate])
    print(f"候補 {i+1}: {sim:.4f}")