import json
import tiktoken
import numpy as np
from collections import defaultdict
import sys

class DatasetAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = self.load_dataset()
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def load_dataset(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def validate_format(self):
        format_errors = defaultdict(int)

        for ex in self.dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1

                if any(
                    k not in ("role", "content", "name", "function_call", "weight")
                    for k in message
                ):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in (
                    "system",
                    "user",
                    "assistant",
                    "function",
                ):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                function_call = message.get("function_call", None)

                if (not content and not function_call) or not isinstance(content, str):
                    format_errors["missing_content"] += 1

            if not any(message.get("role", None) == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

        return format_errors

    def num_tokens_from_messages(self, messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(self, messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(self.encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(self, values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values):.2f}, {np.median(values):.2f}")
        print(f"p5 / p95: {np.quantile(values, 0.05):.2f}, {np.quantile(values, 0.95):.2f}")

    def analyze_data(self):
        n_missing_system = 0
        n_missing_user = 0
        n_messages = []
        convo_lens = []
        assistant_message_lens = []

        for ex in self.dataset:
            messages = ex["messages"]
            if not any(message["role"] == "system" for message in messages):
                n_missing_system += 1
            if not any(message["role"] == "user" for message in messages):
                n_missing_user += 1
            n_messages.append(len(messages))
            convo_lens.append(self.num_tokens_from_messages(messages))
            assistant_message_lens.append(self.num_assistant_tokens_from_messages(messages))

        print("Num examples missing system message:", n_missing_system)
        print("Num examples missing user message:", n_missing_user)
        self.print_distribution(n_messages, "num_messages_per_example")
        self.print_distribution(convo_lens, "num_total_tokens_per_example")
        self.print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
        n_too_long = sum(l > 16385 for l in convo_lens)
        print(
            f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning"
        )

        return convo_lens  # コスト見積もりのために会話の長さを返す

    def estimate_cost(self, convo_lens):
        MAX_TOKENS_PER_EXAMPLE = 16385

        TARGET_EPOCHS = 3
        MIN_TARGET_EXAMPLES = 100
        MAX_TARGET_EXAMPLES = 25000
        MIN_DEFAULT_EPOCHS = 1
        MAX_DEFAULT_EPOCHS = 25

        n_epochs = TARGET_EPOCHS
        n_train_examples = len(self.dataset)
        if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
            n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
        elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
            n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

        n_billing_tokens_in_dataset = sum(
            min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens
        )
        print(
            f"Dataset has ~{n_billing_tokens_in_dataset:,} tokens that will be charged for during training"
        )
        print(f"By default, you'll train for {n_epochs} epochs on this dataset")
        print(
            f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset:,} tokens"
        )

    def run_analysis(self):
        print("Dataset loaded. Validating format...")
        format_errors = self.validate_format()
        if format_errors:
            print("Format errors found:")
            for k, v in format_errors.items():
                print(f"{k}: {v}")
        else:
            print("No format errors found.")

        print("\nAnalyzing data...")
        convo_lens = self.analyze_data()

        print("\nEstimating cost...")
        self.estimate_cost(convo_lens)

def main():
    if len(sys.argv) != 2:
        print("Usage: python prep_and_analisys_dataset.py <path_to_dataset.jsonl>")
        sys.exit(1)

    data_path = sys.argv[1]
    analyzer = DatasetAnalyzer(data_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()