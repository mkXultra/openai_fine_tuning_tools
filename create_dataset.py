from datasets import load_dataset
import json
import os
import sys
from abc import ABC, abstractmethod
import re
from src.lib.embed.labse import LaBSEEmbedder
from prep_and_analisys_dataset import DatasetAnalyzer
from typing import Tuple
from datetime import datetime

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

def load_config(config_file_path):
    with open(config_file_path, "r", encoding="utf-8") as path:
        return json.load(path)

def write_config(config, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as path:
        json.dump(config, path, indent=4, ensure_ascii=False)


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
        self.end_index = 0
        self.config = config
        self.parser = parser
        self.embedder = LaBSEEmbedder()
        self.is_debug = is_debug
        self.similarity = config.get("similarity", 0.9)
        self.japanese_ratio = config.get("japanese_ratio", 0.6)
        self.processed_en = set()

    def log(self, message):
        if self.is_debug:
            pass
            # print(message)
    
    def is_japanese(self, text):
        # 日本語文字（ひらがな、カタカナ、漢字）をカウント
        japanese_chars = re.findall(r'[ぁ-んァ-ン一-龥々]', text)
        
        # テキストの総文字数
        total_chars = len(text)
        
        # 日本語文字の割合を計算
        japanese_ratio = len(japanese_chars) / total_chars if total_chars > 0 else 0
        
        if self.is_debug and not(len(japanese_chars) > 0 and japanese_ratio >= self.japanese_ratio):
            # self.log("len(japanese_chars) > 0", len(japanese_chars) > 0)
            # self.log("japanese_ratio >= self.japanese_ratio", japanese_ratio >= self.japanese_ratio)
            self.log(f"japanese wrong text: {text}")

        # 日本語文字が含まれていて、かつ{self.japanese_ratio}%以上であればTrueを返す
        return len(japanese_chars) > 0 and japanese_ratio >= self.japanese_ratio


    def is_clean_data(self, en, jp) -> bool:
        # check jp is japanese text
        if not self.is_japanese(jp):
            return False
        # check embedding similarity
        similarity = self.embedder.compare_texts(en, jp)
        if self.is_debug:
            self.log(f"similarity: {similarity}")
        return similarity >= self.similarity

    def create_dataset(self, config, output_file, start, limit):
        parser = self.parser
        with open(output_file, "w", encoding="utf-8") as f:
            entries_processed = 0
            for i in range(start, parser.data_length()):
                self.end_index = i
                if entries_processed >= limit:
                    break
                en, jp = parser.parse(i)
                if not self.is_clean_data(en, jp):
                    self.log("not clean data or duplicate en")
                    self.log(f"en: {en}")
                    self.log(f"jp: {jp}")
                    continue
                if en in self.processed_en:
                    self.log("duplicate en")
                    continue
                print(f"Processing entry {entries_processed+1} of {limit}")
                f.write(
                    json.dumps(
                        make_messages(config["system"], config["user"], en, jp),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self.processed_en.add(en)  # 処理したenを追加
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
        print("example: python create_dataset.py prompt_test4.json")
        sys.exit(1)

    config_file_path = sys.argv[1]
    config_file = os.path.splitext(os.path.basename(config_file_path))[0]
    config = load_config(config_file_path)

    config["ft_dataset_file_start_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_config(config, config_file_path)

    # Create output directory
    output_dir = f"config/{config_file}"
    os.makedirs(output_dir, exist_ok=True)

    # Create main dataset
    main_output_file = f"{output_dir}/{config_file}_dataset.jsonl"
    start = config.get("start", 0)
    limit = config.get("limit", 100)
    parser = get_parser(config["dataset"])
    data_maker = DataMaker(config, parser, is_debug=True)
    data_maker.create_dataset(config, main_output_file, start, limit)
    config["ft_dataset_file"] = main_output_file
    config["end_index"] = data_maker.end_index
    config["ft_dataset_file_created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_config(config, config_file_path)
    print(f"config saved to {config_file_path}")
    analyzer = DatasetAnalyzer(main_output_file)
    analyzer.run_analysis()

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
