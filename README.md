# Solid ML Stack

Solid ML Stackは、包括的な機械学習パイプラインを提供するPythonプロジェクトです。Bronze-Silver-Goldアーキテクチャに基づくデータ処理と、複数の機械学習アルゴリズム（XGBoost、CatBoost、LightGBM）を統合した本格的なMLプラットフォームです。

## 🚀 特徴

- **データレイクアーキテクチャ**: Bronze（生データ）→ Silver（クリーニング）→ Gold（特徴量エンジニアリング）の3層構造
- **高性能データ処理**: Polars、DuckDBを使用した高速データ処理
- **多様なMLアルゴリズム**: XGBoost、CatBoost、LightGBM、アンサンブル学習をサポート
- **実験管理**: MLflow統合による実験追跡と再現性の確保
- **自動化**: Makefileによる簡単なコマンド実行
- **型安全性**: 厳密なTypeScript型チェック（mypy）

## 📁 プロジェクト構造

```
solid-ml-stack/
├── README.md                 # プロジェクト概要
├── Makefile                  # ビルド自動化
├── pyproject.toml           # プロジェクト設定
├── requirements.txt         # 依存関係
├── data/                    # データディレクトリ
│   ├── raw/                # 生データ
│   └── dwh/                # データウェアハウス（DuckDB）
├── src/                     # ソースコード
│   ├── data_stage/         # データ処理ステージ
│   │   ├── bronze_stage.py # Bronze層処理
│   │   ├── silver_stage.py # Silver層処理
│   │   └── gold_stage.py   # Gold層処理
│   ├── ml/                 # 機械学習モジュール
│   │   ├── models/         # モデル定義
│   │   ├── training/       # 学習パイプライン
│   │   └── utils/          # MLユーティリティ
│   ├── features/           # 特徴量エンジニアリング
│   ├── pipelines/          # メインパイプライン
│   └── utils/              # 共通ユーティリティ
├── artifacts/              # 生成物（モデル、メトリクス）
├── experiments/            # 実験結果
├── tests/                  # テストスイート
└── docs/                   # ドキュメント
```

## 🛠️ セットアップ

### 前提条件

- Python 3.8以上
- pip

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd solid-ml-stack

# 開発環境セットアップ
make dev-setup

# 依存関係インストール
pip install -r requirements.txt
```

## 🚀 クイックスタート

### 1. データパイプライン実行

```bash
# 完全パイプライン実行（Bronze → Silver → Gold）
make pipeline

# 個別レイヤー実行
make bronze    # Bronze層のみ
make silver    # Silver層のみ
make gold      # Gold層のみ

# パイプライン状態確認
make status
```

### 2. 機械学習モデル実行

```bash
# 個別モデル学習
make ml-xgb    # XGBoost
make ml-cat    # CatBoost
make ml-lgbm   # LightGBM

# アンサンブル学習
make ml-ensemble  # アンサンブル推論
make ml-stack     # スタッキング推論

# 全モデル一括実行
make ml-all

# 実験結果レポート生成
make ml-report
```

### 3. 開発・テスト

```bash
# コードフォーマット
make lint

# テスト実行
make test

# クリーンアップ
make clean
```

## 📊 データパイプライン

### Bronze層（生データ）
- 生CSVデータの読み込み
- 基本的なデータ検証
- DuckDBへの格納

### Silver層（データクリーニング）
- データ型の正規化
- 欠損値処理
- 異常値検出・処理
- データ品質チェック

### Gold層（特徴量エンジニアリング）
- 特徴量生成
- 特徴量選択
- スケーリング
- モデル学習用データセット作成

## 🤖 機械学習機能

### サポートアルゴリズム
- **XGBoost**: 勾配ブースティング決定木
- **CatBoost**: カテゴリカル特徴量に強いブースティング
- **LightGBM**: 高速な勾配ブースティング
- **アンサンブル**: 複数モデルの組み合わせ
- **スタッキング**: メタ学習による予測精度向上

### 実験管理
- MLflow統合による実験追跡
- メトリクス自動保存（RMSE、MAE、R²）
- 特徴量重要度分析
- クロスバリデーション

## 🔧 設定

### 環境変数
```bash
# CatBoost学習ディレクトリ
export CATBOOST_TRAIN_DIR=./artifacts/catboost_info

# データベースパス
export ML_DB=./data/dwh/solid_ml.duckdb
```

### 設定ファイル
- `pyproject.toml`: プロジェクト設定、依存関係
- `requirements.txt`: Python依存関係
- `Makefile`: ビルド・実行コマンド

## 📈 使用例

### データパイプライン実行
```python
from src.pipelines.main import main
from src.utils.config import Config

# 設定
config = Config(db_path="data/dwh/solid_ml.duckdb")
logger = setup_logging("INFO")

# パイプライン実行
main(config, ["bronze", "silver", "gold"], logger)
```

### 機械学習モデル学習
```python
from src.ml.training.train import main

# XGBoostモデル学習
main([
    "--db", "data/dwh/solid_ml.duckdb",
    "--model", "xgb",
    "--target", "price"
])
```

## 🧪 テスト

```bash
# 全テスト実行
pytest

# カバレッジ付きテスト
pytest --cov=src --cov-report=html

# 特定テスト実行
pytest tests/test_bronze_stage.py
```

## 📚 ドキュメント

- [アーキテクチャ設計](docs/architecture.md): プロジェクト構造と設計原則
- [TODO](docs/todo.md): 開発予定機能
- [API仕様](docs/README.md): 詳細なAPIドキュメント

## 🤝 開発ガイドライン

### コーディング規約
- **モジュール・変数・関数**: `snake_case`
- **クラス**: `PascalCase`
- **型ヒント**: 必須（mypy --strict対応）
- **コメント**: 英語で記述

### ブランチ戦略
- `main`: 安定版・デプロイ対象
- `dev`: 開発集約ブランチ
- `feat/*`: 機能開発
- `fix/*`: バグ修正

### 品質保証
- **lint**: black, flake8, mypy
- **テスト**: pytest（カバレッジ90%以上）
- **pre-commit**: コミット前自動チェック

## 📊 パフォーマンス

### データ処理性能
- **Polars**: 高速なDataFrame処理
- **DuckDB**: インメモリ分析データベース
- **並列処理**: マルチコア活用

### 機械学習性能
- **XGBoost**: GPU対応
- **CatBoost**: カテゴリカル特徴量最適化
- **LightGBM**: メモリ効率的な学習

## 🔒 セキュリティ

- 入力データ検証
- SQLインジェクション対策
- 機密情報の環境変数管理
- ログ出力の機密情報除外

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 👥 コントリビューション

1. フォークしてブランチを作成
2. 機能開発・バグ修正
3. テスト追加・実行
4. プルリクエスト作成

## 📞 サポート

- **Issues**: GitHub Issuesでバグ報告・機能要望
- **Discussions**: GitHub Discussionsで質問・議論
- **Wiki**: 詳細な使用方法・トラブルシューティング

## 🔄 更新履歴

### v0.1.0 (2024-01-XX)
- 初期リリース
- Bronze-Silver-Goldデータパイプライン
- XGBoost、CatBoost、LightGBM統合
- MLflow実験管理
- 基本的なMakefile自動化

---

**Solid ML Stack** - 堅牢で拡張可能な機械学習プラットフォーム
