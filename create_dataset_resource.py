import json

def main():
    en_file = 'original/en.txt'
    ja_file = 'original/ja_utf8.txt'
    output_file = 'original_dataset.jsonl'
    
    en_lines = read_file(en_file)
    ja_lines = read_file(ja_file)
    
    if len(en_lines) != len(ja_lines):
        print("Error: Files have different number of lines.")
        return
    
    inserted_count = 0
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for index, (en, ja) in enumerate(zip(en_lines, ja_lines), start=1):
            if "################" in en or "################" in ja:
                continue  # Skip this line if it contains the specified prefix
            if ja.strip() == "" or en.strip() == "":
                continue
            inserted_count += 1
            data = {
                "id": inserted_count,
                "line": index,
                "translation": {
                    "en": en.strip(),
                    "ja": ''.join(ja.strip().split())  # 日本語の文字間のスペースを除去
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
    main()