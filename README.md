# openai_fine_tuning_tools

このリポジトリは、OpenAIのファインチューニングプロセスをサポートするためのツールセットです。主に以下の機能を提供します：

- ファインチューニング用データセットの作成
- ファインチューニングモデルの構築
- モデルの比較

これらのツールを使用することで、OpenAIのモデルを特定のタスクや領域に適応させるプロセスを効率化し、カスタマイズされた高性能なAIモデルの開発をサポートします。

## 使い方

1. セットアップ
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   export OPENAI_API_KEY=your_api_key
   ```
   このスクリプトを実行して、必要な環境をセットアップします。

2. データセットの作成
   ```
   python create_dataset.py <config_name>
   example: python create_dataset.py prompt_test_example
   ```
   このスクリプトを使用して、ファインチューニング用のデータセットを作成します。

3. ファインチューニングモデルの作成
   ```
   python create_fine_tune_model.py <config_name>
   example: python create_fine_tune_model.py prompt_test_example
   ```
   指定した設定ファイルを使用して、ファインチューニングモデルを作成します。
   - ファインチューニングが進行中の場合、スクリプトは完了を待機します。
   - 推定完了時間が5分以上ある場合、スクリプトは終了します。
   - ファインチューニングが完了している場合、設定ファイルにファインチューニングモデル名を追記します。

4. モデルの比較
   ```
   python evaluate_fine_tune_model.py <config_name> <eval_type>
   example: python evaluate_fine_tune_model.py prompt_test_example a
   ```
   ベースモデルとファインチューニングしたモデルの出力を比較します。eval_typeは'a'または'b'を指定し、異なる評価プロンプトを使用します。
   このスクリプトは同じプロンプトを両モデルに与え、その結果を出力して比較を容易にします。

各スクリプトの詳細な使用方法については、それぞれのファイル内のコメントを参照してください。

## 設定ファイル

各プロセスは、JSONフォーマットの設定ファイルを使用して制御します。設定ファイルの例は以下の通りです：

```json
{
  "dataset": "hpprc/alt-parallel-en-ja",
  "system": "You are an expert literary English-Japanese translator specializing in novels and fiction. Please help user to translate",
  "user": "{text}",
  "limit": 1500,
  "start": 0,
  "suffix": "trans-test-v1",
  "base_model": "gpt-4o-mini-2024-07-18",
}
```

主な設定項目：
- `dataset`: 使用するデータセット。Hugging Faceのデータセットを指定できます（例：`"hpprc/alt-parallel-en-ja"`）。また、ローカルのJSONLファイルのパスを指定することも可能です（例：`"./original_dataset.jsonl"`）。
- `system`: ファインチューニング時に使用するシステムプロンプト。モデルの役割や特性を定義します。
- `user`: ユーザープロンプトのテンプレート。`{text}`はデータセットの入力テキストに置き換えられます。
- `limit`: 使用するデータの上限
- `start`: データの開始位置
- `suffix`: モデル名の接尾辞
- `base_model`: ベースとなるモデル

設定ファイルは `<config_name>.json` という名前で保存し、スクリプト実行時に `<config_name>` を指定します。

### 自動追加される情報

`create_fine_tune_model.py`を実行すると、以下の情報が設定ファイルに自動的に追加されます：

- `fine_job_id`: ファインチューニングジョブのID
- `ft_estimated_finish`: ファインチューニングの推定完了時間
- `ft_created_at`: ファインチューニングジョブの作成時間

ファインチューニングが完了すると、以下の情報も追加されます：

- `ft_model`: 作成されたファインチューニングモデルのID
- `used_tokens`: ファインチューニングに使用されたトークン数

これらの情報は、ファインチューニングの進捗管理や結果の追跡に役立ちます。