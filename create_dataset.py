from datasets import load_dataset
import json
# データセットの読み込み

config_file = "prompt_test3"
config_file_path = f"{config_file}.json"
# システムメッセージとプロンプトテンプレートを読み込む
with open(config_file_path, 'r', encoding='utf-8') as path:
    config = json.load(path)

system_message = config['system']
prompt_template = config['user']
dataset_name = config['dataset']
print("dataset name", dataset_name)
dataset = load_dataset(dataset_name)
# データセットの確認
# print("データセットの構造:")
print(dataset)
# minimum 10 example
limit = 20
output_file = f"{config_file}_dataset.jsonl"
# Clean up the output file if it already exists
with open(output_file, "w", encoding="utf-8") as f:
    f.write("")

def make_messages(en, jp):
    prompt = prompt_template.format(text=en)
    return {
        "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
            {"role": "assistant", "content": jp}
        ]
    }

def dataset_parser(_dataset,i):
  if(dataset_name == "hpprc/alt-parallel-en-ja"):
    en = _dataset['train'][i]['en']
    jp = _dataset['train'][i]['ja']
  else:
    en = _dataset['train'][i]['src']
    jp = _dataset['train'][i]['trg']
  return en,jp

    

with open(output_file, "a", encoding="utf-8") as f:
    for i in range(limit):
        print(f"Processing entry {i+1} of {limit}")
        en, jp = dataset_parser(dataset, i)
        f.write(json.dumps(make_messages(en, jp), ensure_ascii=False) + "\n")

print(f"JSONL file '{output_file}' has been created with {len(dataset['train'])} entries.")

# 評価用データセットの作成
eval_limit = 10
eval_output_file = f"{config_file}_evaluation_dataset.jsonl"
eval_output_file_en = f"{config_file}_evaluation_dataset_en.txt"
eval_output_file_jp = f"{config_file}_evaluation_dataset_jp.txt"

with open(eval_output_file, "w", encoding="utf-8") as f:
    for i in range(10, 20):  # インデックス11-20に相当
        print(f"Processing evaluation entry {i+1} of {i+eval_limit}")
        en, jp = dataset_parser(dataset, i)
        f.write(json.dumps(make_messages(en, jp), ensure_ascii=False) + "\n")

print("eval_output_file_en", eval_output_file_en)
with open(eval_output_file_en, "w", encoding="utf-8") as f:
    en, jp = dataset_parser(dataset, 11)
    f.write(en + "\n")

with open(eval_output_file_jp, "w", encoding="utf-8") as f:
    en, jp = dataset_parser(dataset, 11)
    f.write(jp + "\n")

    # for i in range(10,10):  # インデックス11-20に相当
    #     print(f"Processing evaluation entry {i+1} of {i+eval_limit}")
    #     en = dataset['train'][i]['trg']
    #     f.write("-----------------------" + "\n")
    #     f.write(en + "\n")

print(f"Evaluation JSONL file '{eval_output_file}' has been created with {eval_limit} entries.")

# # データセットの統計情報を表示
# print("\nデータセットの統計情報:")
# print(dataset.num_rows)