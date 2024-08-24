from datasets import load_dataset
import json
import os
import sys


def load_config(config_file):
    config_file_path = f"{config_file}.json"
    with open(config_file_path, "r", encoding="utf-8") as path:
        return json.load(path)


def make_messages(system_message, prompt_template, en, jp):
    prompt = prompt_template.format(text=en)
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": jp},
        ]
    }


def dataset_parser(dataset, dataset_name, i):
    if dataset_name == "hpprc/alt-parallel-en-ja":
        en = dataset["train"][i]["en"]
        jp = dataset["train"][i]["ja"]
    elif "original_dataset" in dataset_name:
        print("original_dataset")
        en = dataset["train"][i]["translation"]["en"]
        jp = dataset["train"][i]["translation"]["ja"]
    else:
        en = dataset["train"][i]["src"]
        jp = dataset["train"][i]["trg"]
    return en, jp


def create_dataset(config, dataset, output_file, start, limit):
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(start, min(start + limit, len(dataset["train"]))):
            print(f"Processing entry {i-start+1} of {limit}")
            en, jp = dataset_parser(dataset, config["dataset"], i)
            f.write(
                json.dumps(
                    make_messages(config["system"], config["user"], en, jp),
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"File '{output_file}' has been created with {min(limit, len(dataset['train'])-start)} entries."
    )


def create_single_entry_files(config, dataset, en_file, jp_file, index):
    en, jp = dataset_parser(dataset, config["dataset"], index)
    with open(en_file, "w", encoding="utf-8") as f:
        f.write(en + "\n")
    with open(jp_file, "w", encoding="utf-8") as f:
        f.write(jp + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_dataset.py <config_file>")
        print("example: python create_dataset.py prompt_test4")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    print("Dataset name:", config["dataset"])
    if config["dataset"].endswith(".jsonl"):
        print("jsonl")
        dataset = load_dataset("json", data_files=config["dataset"])
    else:
        dataset = load_dataset(config["dataset"])
    print(dataset)

    # Create output directory
    output_dir = f"config/{config_file}"
    os.makedirs(output_dir, exist_ok=True)

    # Create main dataset
    main_output_file = f"{output_dir}/{config_file}_dataset.jsonl"
    start = config.get("start", 0)
    limit = config.get("limit", 100)
    create_dataset(config, dataset, main_output_file, start, limit)

    # Create evaluation dataset
    eval_output_file = f"{output_dir}/{config_file}_evaluation_dataset.jsonl"
    create_dataset(config, dataset, eval_output_file, 10, 20)

    # Create single entry files for evaluation
    eval_output_file_en = f"{output_dir}/{config_file}_evaluation_dataset_en.txt"
    eval_output_file_jp = f"{output_dir}/{config_file}_evaluation_dataset_jp.txt"
    create_single_entry_files(
        config, dataset, eval_output_file_en, eval_output_file_jp, 11
    )


if __name__ == "__main__":
    main()
