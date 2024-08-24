import json
import argparse

def main(lines_per_dataset, output_file_name):
    en_file = 'original/en.txt'
    ja_file = 'original/ja_utf8.txt'
    output_file = f'{output_file_name}_{lines_per_dataset}.jsonl'
    
    en_lines = read_file(en_file)
    ja_lines = read_file(ja_file)
    
    if len(en_lines) != len(ja_lines):
        print("Error: Files have different number of lines.")
        return
    
    inserted_count = 0
    buffer = []
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for index, (en, ja) in enumerate(zip(en_lines, ja_lines), start=1):
            if "################" in en or "################" in ja:
                if buffer:
                    write_data(jsonl_file, buffer, inserted_count)
                    inserted_count += 1
                    buffer = []
                continue
            if ja.strip() == "" or en.strip() == "":
                continue
            buffer.append((index, en.strip(), ''.join(ja.strip().split())))
            if len(buffer) == lines_per_dataset:
                write_data(jsonl_file, buffer, inserted_count)
                inserted_count += 1
                buffer = []
        
        # 残りのデータを処理
        if buffer:
            write_data(jsonl_file, buffer, inserted_count)

def write_data(jsonl_file, buffer, inserted_count):
    data = {
        "id": inserted_count + 1,
        "lines": [line[0] for line in buffer],
        "translation": {
            "en": "\n".join([line[1] for line in buffer]),
            "ja": "\n".join([line[2] for line in buffer])
        }
    }
    json.dump(data, jsonl_file, ensure_ascii=False)
    jsonl_file.write('\n')

def read_file(filename):
    encodings = ['utf-8', 'shift_jis', 'euc_jp', 'iso2022_jp']
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as file:
                return file.readlines()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode the file {filename} with any of the attempted encodings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset resource')
    parser.add_argument('--lines', type=int, required=True, help='Number of lines per dataset')
    parser.add_argument('--output', type=str, required=True, help='Output file name without extension')
    args = parser.parse_args()

    main(args.lines, args.output)