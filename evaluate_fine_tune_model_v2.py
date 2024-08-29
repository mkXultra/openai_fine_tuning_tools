import sys
import json
from enum import Enum
from openai import OpenAI
from typing import Dict, List

class EvalType(Enum):
    A = "a"
    B = "b"
    C = "c"

class Config:
    def __init__(self, config_file: str):
        with open(f"{config_file}.json", "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
    def get(self, key: str, default=None):
        return self.data.get(key, default)

class EvaluationRunner:
    def __init__(self, config: Config, eval_type: EvalType):
        self.config = config
        self.eval_type = eval_type
        self.client = OpenAI()

    def run(self):
        target_str = self.load_target_string()
        messages = self.make_messages(target_str)
        
        print_completion("original", target_str)
        
        models = self.get_models_to_evaluate()
        completions = self.get_completions(models, messages)
        
        for model, completion in completions.items():
            print_completion(f"{model} case", get_completion_text(completion))
        
        self.print_custom_case()

    def load_target_string(self) -> str:
        eval_files = {
            EvalType.A: "evaluation.txt",
            EvalType.B: "evaluation_dataset.txt",
            EvalType.C: "evaluation_dataset2.txt"
        }
        with open(eval_files[self.eval_type], "r", encoding="utf-8") as f:
            return f.read()

    def make_messages(self, text: str) -> List[Dict[str, str]]:
        system_message = self.config.get("system")
        prompt_template = self.config.get("user")
        prompt = prompt_template.format(text=text)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

    def get_models_to_evaluate(self) -> List[str]:
        models = [self.config.get("base_model"), self.config.get("ft_model")]
        for i in range(1, 4):
            model = self.config.get(f"compare_model{i if i > 1 else ''}")
            if model and (self.eval_type != EvalType.A or i < 3):
                models.append(model)
        return [model for model in models if model]

    def get_completions(self, models: List[str], messages: List[Dict[str, str]]) -> Dict[str, dict]:
        return {model: get_completion(self.client, model, messages) for model in models}

    def print_custom_case(self):
        custom_files = {
            EvalType.A: "evaluation_gpt4.txt",
            EvalType.B: "evaluation_dataset_ja.txt",
            EvalType.C: "evaluation_dataset2_ja.txt"
        }
        custom_name = "gpt4o" if self.eval_type == EvalType.A else "custom"
        with open(custom_files[self.eval_type], "r", encoding="utf-8") as f:
            print_completion(f"{custom_name} case", f.read())

def get_completion(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> dict:
    return client.chat.completions.create(model=model, messages=messages)

def get_completion_text(completion: dict) -> str:
    return completion.choices[0].message.content

def print_completion(model_name: str, text: str):
    print(f"# {model_name}:")
    print("```")
    print(text)
    print("```")
    print("\n")

def main(config_file: str, eval_type: str):
    config = Config(config_file)
    try:
        eval_type_enum = EvalType(eval_type)
    except ValueError:
        raise ValueError("Invalid eval_type. Use 'a', 'b', or 'c'.")
    
    runner = EvaluationRunner(config, eval_type_enum)
    runner.run()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_fine_tune_model.py <config_file> <eval_type>")
        print("eval_type should be 'a', 'b', or 'c'")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
