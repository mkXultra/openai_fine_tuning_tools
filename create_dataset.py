from datasets import load_dataset
import json
import os
import sys
from abc import ABC, abstractmethod
import re
from src.lib.embed.labse import LaBSEEmbedder
from typing import Tuple

class DatasetParser(ABC):
    @abstractmethod
    def __init__(self, dataset_name):
        pass

    @abstractmethod
    def parse(self, index):
        pass

    @abstractmethod
    def data_length(self):
        pass

class AltParallelEnJaParser(DatasetParser):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.dataset = load_dataset(dataset_name)

    def parse(self, index):
        return self.dataset["train"][index]["en"], self.dataset["train"][index]["ja"]

    def data_length(self):
        return len(self.dataset["train"])

class CcMatrixParser(DatasetParser):
    key = "translation"
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.dataset = load_dataset(dataset_name, "en-ja", split='train')
        # self.dataset = load_dataset(dataset_name, "en-ja", split='train', streaming=True)

    def parse(self, index) -> Tuple[str, str]:
        en = self.dataset[index][self.key]["en"]
        # clean ja
        ja = self.dataset[index][self.key]["ja"].replace("\\n", "").replace("\\", "")
        ja = re.sub(r'\s+', '', ja)  # Remove all whitespace between Japanese characters
        return en, ja

    def data_length(self):
        return self.dataset.num_rows
    
class OriginalDatasetParser(DatasetParser):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.dataset = load_dataset("json", data_files=dataset_name)

    def parse(self, index):
        return self.dataset["train"][index]["translation"]["en"], self.dataset["train"][index]["translation"]["ja"]
    
    def data_length(self):
        return len(self.dataset["train"])

class DefaultParser(DatasetParser):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.dataset = load_dataset(dataset_name)

    def parse(self, index):
        return self.dataset["train"][index]["src"], self.dataset["train"][index]["trg"]

    def data_length(self):
        return len(self.dataset["train"])

def get_parser(dataset_name):
    if dataset_name == "hpprc/alt-parallel-en-ja":
        return AltParallelEnJaParser(dataset_name)
    elif dataset_name == "yhavinga/ccmatrix":
        return CcMatrixParser(dataset_name)
    elif "original_dataset" in dataset_name:
        return OriginalDatasetParser(dataset_name)
    else:
        raise ValueError(f"not supported dataset name: {dataset_name}")

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


class DataMaker:
    def __init__(self, config, parser, is_debug=False):
        self.config = config
        self.parser = parser
        self.embedder = LaBSEEmbedder()
        self.is_debug = is_debug
        self.similarity = config.get("similarity", 0.9)
        self.japanese_ratio = config.get("japanese_ratio", 0.5)
    
    def is_japanese(self, text):
        # 日本語文字（ひらがな、カタカナ、漢字）をカウント
        japanese_chars = re.findall(r'[ぁ-んァ-ン一-龥々]', text)
        
        # テキストの総文字数
        total_chars = len(text)
        
        # 日本語文字の割合を計算
        japanese_ratio = len(japanese_chars) / total_chars if total_chars > 0 else 0
        
        # 日本語文字が含まれていて、かつ50%以上であればTrueを返す
        if self.is_debug and japanese_ratio <= self.japanese_ratio:
            print(f"japanese wrong text: {text}")
        return len(japanese_chars) > 0 and japanese_ratio >= self.japanese_ratio


    def is_clean_data(self, en, jp) -> bool:
        # check jp is japanese text
        if not self.is_japanese(jp):
            return False
        # check embedding similarity
        similarity = self.embedder.compare_texts(en, jp)
        if self.is_debug:
            print(f"similarity: {similarity}")
        return similarity > 0.9

    def create_dataset(self, config, output_file, start, limit):
        parser = self.parser
        with open(output_file, "w", encoding="utf-8") as f:
            entries_processed = 0
            for i in range(start, parser.data_length()):
                if entries_processed >= limit:
                    break
                print(f"Processing entry {entries_processed+1} of {limit}")
                en, jp = parser.parse(i)
                if not self.is_clean_data(en, jp):
                    # if self.is_debug:
                        # print("not clean data")
                        # print(f"en: {en}")
                        # print(f"jp: {jp}")
                    continue
                f.write(
                    json.dumps(
                        make_messages(config["system"], config["user"], en, jp),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                entries_processed += 1
        print(
            f"File '{output_file}' has been created with {entries_processed} entries."
        )
    



# def create_single_entry_files(config, en_file, jp_file, index):
#     parser = get_parser(config["dataset"])
#     en, jp = parser.parse(index)
#     with open(en_file, "w", encoding="utf-8") as f:
#         f.write(en + "\n")
#     with open(jp_file, "w", encoding="utf-8") as f:
#         f.write(jp + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_dataset.py <config_file>")
        print("example: python create_dataset.py prompt_test4")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    # Create output directory
    output_dir = f"config/{config_file}"
    os.makedirs(output_dir, exist_ok=True)

    # Create main dataset
    main_output_file = f"{output_dir}/{config_file}_dataset.jsonl"
    start = config.get("start", 0)
    limit = config.get("limit", 100)
    parser = get_parser(config["dataset"])
    data_maker = DataMaker(config, parser)
    data_maker.create_dataset(config, main_output_file, start, limit)

    # # Create evaluation dataset
    # eval_output_file = f"{output_dir}/{config_file}_evaluation_dataset.jsonl"
    # create_dataset(config, eval_output_file, 10, 20)

    # # Create single entry files for evaluation
    # eval_output_file_en = f"{output_dir}/{config_file}_evaluation_dataset_en.txt"
    # eval_output_file_jp = f"{output_dir}/{config_file}_evaluation_dataset_jp.txt"
    # create_single_entry_files(
    #     config, eval_output_file_en, eval_output_file_jp, 11
    # )


if __name__ == "__main__":
    main()
