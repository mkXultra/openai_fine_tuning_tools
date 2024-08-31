import sys
import json
from openai import OpenAI
from anthropic import Anthropic
from typing import Dict, List, Union
import numpy as np
from datetime import datetime

EMBEDDING_MODEL = "text-embedding-3-large"

class Config:
    def __init__(self, config_file: str):
        with open(config_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
    def get(self, key: str, default=None):
        return self.data.get(key, default)

class EvaluationRunner:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI()
        self.anthropic_client = Anthropic()
    def run(self):
        target_pairs = self.load_target_strings()
        models = self.get_models_to_evaluate()
        model_similarities = {model: {"scores":[], "avg":0, "data":[]} for model in models}
        
        for model in models:
            for epoch in range(self.config.get("epoch", 1)):
                for target_pair in target_pairs:
                    en_text = target_pair["en"]
                    ja_text = target_pair["ja"]
                    messages = self.make_messages(en_text, model)
                    completion = get_completion(self.client, self.anthropic_client, model, messages)
                    try:
                        similarity = self.evaluate(ja_text, get_completion_text(completion))
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Model: {model}, reference: {ja_text}, Completion: {completion}")
                        continue
                    model_similarities[model]["scores"].append(similarity)
                    model_similarities[model]["data"].append(get_completion_text(completion))
                    print(f"Model: {model}, Embedding 類似度: {similarity:.4f}")

        # Calculate average similarities and sort models
        for model, similarities in model_similarities.items():
            avg_similarity = np.mean(similarities["scores"])
            model_similarities[model]["avg"] = avg_similarity
        sorted_models = sorted(model_similarities.items(), key=lambda x: x[1]["avg"], reverse=True)
        
        # Print results
        for model, similarity_data in sorted_models:
            print(f"{model}: {similarity_data['avg']:.4f}")
        
        # Write results to JSON file
        self.write_results_to_json(sorted_models)

    def evaluate(self, reference: str, candidate: str) -> float:
        similarity = self.cosine_similarity(self.get_embedding(reference), self.get_embedding(candidate))
        return similarity

    def get_embedding(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def load_target_strings(self) -> List[Dict[str, str]]:
        with open(self.config.get("dataset"), "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def make_messages(self, text: str, model: str) -> List[Dict[str, str]]:
        if model.startswith("claude"):
            system_message = self.config.get("system")
            prompt_template = self.config.get("user")
            prompt = prompt_template.format(text=text)
        else:
            system_message = self.config.get("system")
            prompt_template = self.config.get("user")
            prompt = prompt_template.format(text=text)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

    def get_models_to_evaluate(self) -> List[str]:
        return self.config.get("models")

    def write_results_to_json(self, model_similarities: Dict[str, Dict[str, Union[List[float], float]]]):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.config.get("output_file", f"evaluation_results_{current_time}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(model_similarities, f, indent=2, ensure_ascii=False)
        print(f"Results written to {output_file}")

def get_completion(client: OpenAI, anthropic_client: Anthropic, model: str, messages: List[Dict[str, str]]) -> dict:
    if model.startswith("claude"):
        system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
        messages = [msg for msg in messages if msg['role'] != 'system']
        if system_message:
            return anthropic_client.messages.create(
                model=model,
                messages=messages,
                system=system_message,
                max_tokens=4096
            )
        return anthropic_client.messages.create(model=model, messages=messages, max_tokens=4096)
    else:
        return client.chat.completions.create(model=model, messages=messages)

def get_completion_text(completion: dict) -> str:
    if hasattr(completion, 'choices'):
        return completion.choices[0].message.content
    elif hasattr(completion, 'content'):
        return completion.content[0].text
    else:
        raise ValueError("No completion text found")

def main(config_file: str):
    config = Config(config_file)
    
    runner = EvaluationRunner(config)
    runner.run()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_fine_tune_model.py <config_file>")
        sys.exit(1)

    main(sys.argv[1])
