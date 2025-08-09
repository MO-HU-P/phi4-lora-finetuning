
# Phi-4 LoRA Fine-tuning with Unsloth 操作マニュアル

このプロジェクトは、UnslothライブラリーとLoRA（Low-Rank Adaptation）を使用してPhi-4モデルを教師ありファインチューニング（Supervised Fine-Tuning）するための操作ガイドです。

## プロジェクト概要

- **ベースモデル**: `unsloth/Phi-4-unsloth-bnb-4bit`
- **データセット**: `msfm/ichikara-instruction-all`
- **出力形式**: LoRAアダプター
- **最適化**: 4bit量子化、LoRA、勾配チェックポイント

## セットアップ手順

### 1. 環境準備

```bash
# プロジェクトディレクトリ移動
cd phi4-finetune

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows

# 依存関係インストール
pip install -r requirements.txt
```

### 2. ファインチューニング実行

```bash
python phi4_lora_finetune.py
```

## 出力ファイル

実行完了後、以下が生成されます：

```
phi4-finetune/
├── lora_model/                         # LoRAアダプター（重要）
│   ├── adapter_model.safetensors       # LoRA重み（最重要）
│   ├── adapter_config.json             # LoRA設定
│   ├── tokenizer.json                  # トークナイザー
│   ├── tokenizer_config.json           # トークナイザー設定
│   ├── special_tokens_map.json         # 特殊トークン
│   ├── vocab.json                      # 語彙辞書
│   ├── merges.txt                      # BPEマージルール
│   └── chat_template.jinja             # チャットテンプレート
├── outputs/                            # 訓練ログ・チェックポイント(checkpoint-*)
└── unsloth_compiled_cache/             # Pythonモジュールが保存されたキャッシュフォルダ
```

## 再トレーニング手順

パラメータを変更して再実行する場合：

### 削除するファイル
```bash
# 前回の結果を削除
rm -rf lora_model/
rm -rf outputs/

# キャッシュ・一時ファイルを削除
rm -rf unsloth_compiled_cache/
```

### 保持するファイル
- `phi4_lora_finetune.py` - メインスクリプト
- `requirements.txt` - 依存関係
- `venv/` - 仮想環境

**削除後の状態**

```
phi4-finetune/
├── phi4_lora_finetune.py
├── requirements.txt
└── venv/
```

## ベンチマークテストデータを使った文書生成

ベースモデルとLoRAモデルを使って、データセット「ELYZA-tasks-100」のinputから、文書生成するためのpythonファイルです。

ファインチューニング実行後には、上記のとおり各種ディレクトリが作成されます。
プロジェクトディレクトリ（phi4-finetune）内に、以下のpythonファイルを実行します。

```bash
python run_benchmark.py
```

すべてのタスクが終了すると、「result_benchmark_*.json」がプロジェクトディレクトリ内に生成されます。
ELYZA-tasks-100には、100件のinput（質問）、output（模範回答）、eval_aspect（評価方法）のデータセットになっています。

main関数のmax_tasksの数値を指定できます。

```python
benchmark.run_benchmark(dataset, max_tasks=100)
```


<br>
<br>

# Phi-4 LoRA ファインチューニング評価システム

Phi-4モデルのLoRAファインチューニング後に、ベースモデルとLoRAモデルの文書生成能力を比較評価するシステムです。

## 概要

このシステムは以下の処理を自動実行します：

1. **ファインチューニング**: Phi-4モデルをLoRAでファインチューニング
2. **ベンチマークテスト**: ELYZA-tasks-100データセットを使用して両モデルで文書生成
3. **比較評価**: 生成された文書をAIエージェントが評価し、どちらが優秀かを判定

## 使用方法

### 1. システム起動

cdコマンドでディレクトリ（phi4_loRA_evaluation）に移動してください。

```bash
cd phi4_loRA_evaluation
```

```bash
docker compose up -d --build
```
最初は、モデルのpullに時間がかかる場合があります。
正常にコンテナがビルドされてもコンテナが起動しない場合は、再起動してみてください。

```bash
docker compose down
docker compose up -d
```

```bash
docker compose restart
```

コンテナが立ち上がりシステムが正常に起動すると、タスク処理が自動開始されます。


### 2. 処理状況の確認

ターミナルでは何も表示されませんが、以下のコマンドで処理状況を確認できます：

```bash
docker compose logs -f langgraph-evaluator
```

### 3. 結果の確認

処理完了後、`output/` フォルダに評価レポートが生成されます。

### 4. システム停止

```bash
docker compose down
```

## ディレクトリ構成

```
phi4_loRA_evaluation/
├── input/           # ベンチマークファイル(result_benchmark_*.json)配置場所
├── output/          # 評価結果出力先
├── config/          # 設定ファイル
├── src/             # メインプログラム
└── docker-compose.yml
```

## 手動実行オプション

コンテナ内での手動実行が可能です：

```bash
# 基本実行
python src/main.py

# カスタム設定
python src/main.py --input-dir /app/input --output-dir /app/output

# 並行処理数を指定（GPUメモリに注意）
python src/main.py --max-concurrent 2

# 中間結果を保存しない
python src/main.py --no-intermediate

# 詳細ログ表示
python src/main.py --verbose

# 簡略ログ表示
python src/main.py --quiet

# 特定のベンチマークファイル指定
python src/main.py --input "result_benchmark_20250101.json"
```

## 主な機能

- **自動モデル比較**: ベースモデル vs LoRAモデル
- **多角的評価**: 複数のAIエージェントによる評価
- **詳細レポート**: 評価結果の詳細分析
- **リソース管理**: GPU使用量の最適化
- **ログ管理**: 実行状況の詳細記録

## 結果の分析・可視化

評価完了後、`BenchmarkAnalyzer.py`を使用して結果を分析できます：

```bash
python BenchmarkAnalyzer.py
```

### 分析機能

- **統計データの表示**: 勝率、改善度の数値分析
- **グラフ可視化**: 評価結果を分かりやすいグラフで表示
- **詳細レポート**: 評価観点別の詳細分析

### カスタマイズ

ファイルパスを変更する場合は、`BenchmarkAnalyzer.py`の366行目を編集：

```python
# デフォルト（自動で最新ファイルを検索）
analyzer = BenchmarkAnalyzer('./output/assessment_report_*.json')

# 特定ファイルを指定する場合
analyzer = BenchmarkAnalyzer('./output/assessment_report_20250101.json')
```

生成される `benchmark_analysis.png` で結果をグラフィカルに確認できます。

## 注意事項

- GPU環境が必要です
- 初回実行時はモデルダウンロードに時間がかかります
- `--max-concurrent > 3` はGPUメモリ不足を起こす可能性があります