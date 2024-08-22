from datasets import load_dataset
import json
import os
import sys

def load_config(config_file):
    config_file_path = f"{config_file}.json"
    with open(config_file_path, 'r', encoding='utf-8') as path:
        return json.load(path)

def make_messages(system_message, prompt_template, en, jp):
    prompt = prompt_template.format(text=en)
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": jp}
        ]
    }

def dataset_parser(dataset, dataset_name, i):
    if dataset_name == "hpprc/alt-parallel-en-ja":
        en = dataset['train'][i]['en']
        jp = dataset['train'][i]['ja']
    else:
        en = dataset['train'][i]['src']
        jp = dataset['train'][i]['trg']
    return en, jp

def create_dataset(config, dataset, output_file, start, end):
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(start, end):
            print(f"Processing entry {i+1} of {end}")
            en, jp = dataset_parser(dataset, config['dataset'], i)
            f.write(json.dumps(make_messages(config['system'], config['user'], en, jp), ensure_ascii=False) + "\n")
    print(f"File '{output_file}' has been created with {end - start} entries.")

def create_single_entry_files(config, dataset, en_file, jp_file, index):
    en, jp = dataset_parser(dataset, config['dataset'], index)
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
    
    print("Dataset name:", config['dataset'])
    dataset = load_dataset(config['dataset'])
    print(dataset)

    # Create output directory
    output_dir = f"config/{config_file}"
    os.makedirs(output_dir, exist_ok=True)

    # Create main dataset
    main_output_file = f"{output_dir}/{config_file}_dataset.jsonl"
    create_dataset(config, dataset, main_output_file, 0, 20)

    # Create evaluation dataset
    eval_output_file = f"{output_dir}/{config_file}_evaluation_dataset.jsonl"
    create_dataset(config, dataset, eval_output_file, 10, 20)

    # Create single entry files for evaluation
    eval_output_file_en = f"{output_dir}/{config_file}_evaluation_dataset_en.txt"
    eval_output_file_jp = f"{output_dir}/{config_file}_evaluation_dataset_jp.txt"
    create_single_entry_files(config, dataset, eval_output_file_en, eval_output_file_jp, 11)

if __name__ == "__main__":
    main()