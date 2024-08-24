import sys
import json
from openai import OpenAI


def load_config(config_file):
    config_file_path = f"{config_file}.json"
    with open(config_file_path, "r", encoding="utf-8") as path:
        return json.load(path)


def load_target_string(target_file):
    eval_output_file = target_file
    # eval_output_file = "evaluation_dataset.txt"
    # eval_output_file = f"evaluation.txt"
    with open(eval_output_file, "r", encoding="utf-8") as f:
        return f.read()


def make_messages(system_message, prompt_template, text):
    prompt = prompt_template.format(text=text)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]


def get_completion(client, model, messages):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion


def get_completion_text(completion):
    return completion.choices[0].message.content


def print_completion(model_name, text):
    print(f"# {model_name}:")
    print("```")
    print(text)
    print("```")
    print("\n")


def main(config_file, eval_type):
    config = load_config(config_file)

    if eval_type == "a":
        eval_output_file = "evaluation.txt"
    elif eval_type == "b":
        eval_output_file = "evaluation_dataset.txt"
    elif eval_type == "c":
        eval_output_file = "evaluation_dataset2.txt"
    else:
        raise ValueError("Invalid eval_type. Use 'a' or 'b'.")

    target_str = load_target_string(eval_output_file)

    system_message = config["system"]
    prompt_template = config["user"]

    messages = make_messages(system_message, prompt_template, target_str)

    client = OpenAI()  # API key should be set as an environment variable

    print_completion("original", target_str)

    base_completion = get_completion(client, config["base_model"], messages)
    print_completion(
        config["base_model"] + "case", get_completion_text(base_completion)
    )
    ft_model = config.get("ft_model")
    if not ft_model:
        print("ft_model is not set in the config file.")
    ft_completion = get_completion(client, ft_model, messages)
    print_completion(ft_model + " case", get_completion_text(ft_completion))

    compare_model = config.get("compare_model")
    if compare_model:
        compare_completion = get_completion(client, compare_model, messages)
        print_completion(
            compare_model + " case", get_completion_text(compare_completion)
        )

    compare_model2 = config.get("compare_model2")
    if compare_model2:
        compare_completion2 = get_completion(client, compare_model2, messages)
        print_completion(
            compare_model2 + " case", get_completion_text(compare_completion2)
        )

    compare_model3 = config.get("compare_model3")
    if compare_model3 and eval_type != "a":
        compare_completion3 = get_completion(client, compare_model3, messages)
        print_completion(
            compare_model3 + " case", get_completion_text(compare_completion3)
        )

    if eval_type == "a":
        with open("evaluation_gpt4.txt", "r", encoding="utf-8") as f:
            print_completion("gpt4o case", f.read())
    if eval_type == "b":
        with open("evaluation_dataset_ja.txt", "r", encoding="utf-8") as f:
            print_completion("custom case", f.read())
    if eval_type == "c":
        with open("evaluation_dataset2_ja.txt", "r", encoding="utf-8") as f:
            print_completion("custom case", f.read())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_fine_tune_model.py <config_file> <eval_type>")
        print("eval_type should be 'a' or 'b'")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
