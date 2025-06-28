# solid_ml - プロジェクト構成リファクタリングガイド

*(2025-06-28 版)*

このドキュメントは **`solid_ml`** リポジトリを "再現性・可読性・保守性" の三拍子そろった ML プロジェクトに仕立てるためのフォルダ／ファイル構成指針をまとめたものです。
そのまま **`docs/architecture.md`** などにコミットしておけば、今後の開発メンバーも迷わず参加できます。

---

### 1. ディレクトリ全体像

```
solid_ml/                 ← Git リポジトリ (= プロジェクトルート)
├── README.md             ← 3 分で動かせる最小手順を必ず掲載
├── Makefile              ← make lint / test / train / clean…
├── pyproject.toml        ← Poetry or PEP-621 (任意) ※requirements 方式なら不要
├── .gitignore            ← __pycache__, *.pkl, artifacts/, experiments/ を除外
│
├── data/                 ← 実データ (追跡は DVC or Git-LFS 推奨)
│   ├── raw/              : 生データ (変更禁止)
│   ├── interim/          : 一時加工 (Bronze 相当)
│   ├── processed/        : モデル入力確定版 (Gold 相当)
│   └── external/         : 配布不可・契約データなど
│
├── artifacts/            ← 特徴量 / モデル / スケーラー等 (.pkl, .onnx, .json)
│   └── features/
│
├── experiments/          ← 1 実験 = 1 フォルダ (metrics.json, params.yaml, logs/)
│   └── 2025-06-28T15-13-10/
│
├── docs/                 ← 設計・仕様・ADR・講習メモなど
│   └── architecture.md   : ← 本ファイルを置く想定
│
└── src/                  ← ★PYTHONPATH の起点 (import ルート)
    ├── data_stage/       : Bronze / Silver / Gold 各処理
    │   ├── __init__.py
    │   ├── bronze_stage.py
    │   ├── silver_stage.py
    │   └── gold_stage.py
    │
    ├── features/         : 特徴量生成
    │   ├── __init__.py
    │   └── feature_builder.py
    │
    ├── models/           : モデルクラス & 推論ラッパ
    │   ├── __init__.py
    │   ├── cat_model.py
    │   ├── lgbm_model.py
    │   ├── xgb_model.py
    │   └── ensemble_model.py
    │
    ├── pipelines/        : 学習・推論パイプライン
    │   ├── __init__.py
    │   ├── train.py
    │   └── predict.py
    │
    ├── utils/            : 共通ユーティリティ
    │   ├── __init__.py
    │   ├── io.py
    │   └── logging.py
    │
    └── __init__.py       : 空で OK (名前空間用)
tests/                    ← pytest ベースのユニット / 結合テスト
```

---

### 2. 各ディレクトリの責務

| 階層               | 何を置くのか / 置かないのか                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| **data/**        | *置く* : CSV・Parquet・画像など"真のソース"データ<br>*置かない* : 中間特徴量・モデル・ログ                                           |
| **artifacts/**   | 学習後に生成される **派生物**。タイムスタンプを付けて衝突回避しつつ、容量課題は DVC or LFS で管理                                            |
| **experiments/** | "1 回の train/predict 実行" をまるごと封入：<br>`metrics.json` / `params.yaml` / `stdout.log` / `catboost_info/` |
| **src/**         | Python コード専用。**サブパッケージ名＝担当レイヤを示す英小文字** を原則とする (例: `data_stage`, `models`)                            |
| **tests/**       | ディレクトリ構造は **src と鏡写し** がベスト (`tests/data_stage/test_bronze.py` など)                                   |
| **docs/**        | README より詳細な設計／運用／ADR。Swagger 仕様書などはここ or `src/pipelines/openapi/`                                   |

---

### 3. コーディング規約 & 命名ルール

* **モジュール / 変数 / 関数**: `snake_case`
* **クラス**: `PascalCase`
* **データクラス**: `SomethingConfig`, `SomethingParams`
* **実験ディレクトリ**: `YYYY-MM-DDThh-mm-ss` (ISO 8601, タイムゾーン無し)
* **型ヒント**: Python 3.12 の `|` union を活用し、`mypy --strict` が通ること

---

### 4. 開発フロー (推奨)

1. **ブランチ戦略**
   `main` ← 安定 / デプロイ対象
   `dev`  ← 開発集約
   `feat/*`, `fix/*` でトピックを切り、PR 対象は `dev`
2. **pre-commit**

   * black / ruff / isort / detect-secrets
   * commit 前自動フォーマット
3. **CI (GitHub Actions 例)**

   * lint → pytest → (optional) build-docker → push-ECR
4. **実験**

   ```bash
   poetry run python -m pipelines.train --config configs/xgb.yaml
   # 実験後 artifacts/, experiments/ に自動書き出し
   ```
5. **再現**

   * `experiments/<run>/params.yaml` を指定し再学習 → 指標一致を確認
6. **本番化**

   * `.onnx` / `catboost.cbm` を `artifacts/models/v1/` へ固定タグ → Docker build

---

### 5. 既存リポジトリからの移行手順

| ステップ                        | コマンド例                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------- |
| 1. ブランチ作成                   | `git checkout -b refactor/layout`                                                   |
| 2. フォルダ移動                   | `git mv src/bronze.py src/data_stage/bronze_stage.py` … (一覧は後述)                     |
| 3. 不要物削除                    | `find . -name '__pycache__' -exec rm -r {} +`                                       |
| 4. import 修正                | VS Code Rename, または `sed -i 's/from src.ml./from models./g' $(git ls-files '*.py')` |
| 5. テスト更新                    | `pytest -q` が green になるまで修正                                                         |
| 6. ドキュメント反映                 | 本ファイルを `docs/architecture.md` として追加                                                 |
| 7. PR → レビュー → squash merge |                                                                                     |

> **注意**: 旧 `solid-ml-stack/src/ml/` と `solid-ml-stack/solid-ml-stack/src/ml/` は**完全に削除**してからインポート経路を一本化すること。

---

### 6. よくある質問 (FAQ)

**Q. `src/` 配下にさらに `solid_ml/` パッケージを作らなくて大丈夫？**
A. はい。"リポジトリ名とパッケージ名が衝突"問題を避けるため、今回は **機能別サブパッケージ** 直置きに統一しました。`PYTHONPATH=./src` だけ通せば `import data_stage` 等で利用できます。

**Q. Poetry を採用しない場合の依存管理は？**
A. `requirements/base.txt`, `requirements/dev.txt` を `pip-tools` で管理し、CI では `pip-sync requirements/dev.txt` を実行する形がシンプルです。

**Q. 実験ディレクトリを Git で追わないと履歴が切れるのでは？**
A. 実験は **量が膨大かつバイナリ混在** のため、Git 管理に向きません。

* 軽量メタデータ (metrics.json, params.yaml) をコミット
* 重量ファイル (モデル, feature.pkl) は DVC/LFS で外部ストレージ連携
  という二段槈えがベターです。

---

### 7. 仕上げチェックリスト

* [ ] `poetry install` → `pytest -q` が通る
* [ ] `make lint` で警告ゼロ
* [ ] `python -m pipelines.train` が artifacts/ と experiments/ を正しく生成
* [ ] README の "Quick Start" がコピペで完走
* [ ] Docker build (`docker build .`) → `CMD ["python","-m","pipelines.predict"]` が動く

---

以上が **solid_ml リファクタリング完全版** です。
「`git mv` コマンドリストをもっと細かく」「CI YAML を丸ごと書いてほしい」など、追加のリクエストがあればお気軽にどうぞ。 