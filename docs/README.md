# ミニマル & ソリッド ML スタック  ― Kaggle 銅メダル専用

> **目的**: Kaggle コンペで銅メダルを “個人” で獲得する。そのために **環境構築に 1 秒も費やさず、特徴量設計・モデル改善に 100 % 集中** できる構成を示す。

---

## 1. コンセプト（削ぎ落としの原則）

| やらない                       | 代わりに採用                         | 理由                   |
| -------------------------- | ------------------------------ | -------------------- |
| Docker / Poetry / Conda    | **グローバル pip** (WSL Ubuntu)     | 依存が少なく、セットアップが秒速で終わる |
| ノートブック                     | **純 Python スクリプト + MLflow UI** | 実行順ズレ・セル副作用を排除し再現性◎  |
| クラウド CI / GPU 移送           | **手元 GPU / CPU 全力回し**          | 単独参戦ではローカルが最速        |
| 仮想環境 (pyenv / venv / pipx) | 必要になったら後付け                     | Kaggle だけなら衝突リスク小    |

---

## 2. ミニマム・ツールセット & インストール

| レイヤ       | ツール                           | 1 行インストール                                   | 概要                                   |
| --------- | ----------------------------- | ------------------------------------------- | ------------------------------------ |
| 実験管理      | **MLflow 2.x**                | `pip install mlflow`                        | 実行ログとモデルを自動保存。`mlflow ui` でブラウザ比較    |
| ETL       | **polars**                    | `pip install polars[all]`                   | pandas の 3–10 倍速。lazy + streaming 対応 |
| モデル       | LightGBM / CatBoost / XGBoost | `pip install lightgbm catboost xgboost`     | 汎用 Tabular GBDT 3 兄弟。GPU 切替自在        |
| ハイパラ (任意) | Optuna                        | `pip install optuna[lightgbm]`              | 重み探索やパラメータ最適化に使用                     |
| DB (任意)   | DuckDB                        | `pip install duckdb`                        | OOF 結合やメタ解析用。SQL で瞬時に集計              |
| 可視化 (任意)  | Metabase                      | `docker run -p 3000:3000 metabase/metabase` | 投稿や社内共有用。コンペ序盤は不要                    |

**セットアップ例 (WSL 上)**

```
# 事前にビルド系ライブラリを入れておくと失敗しにくい
sudo apt update
sudo apt install -y build-essential python3-dev libffi-dev libssl-dev \
                    libblas-dev liblapack-dev gfortran

# PEP 668 を回避して一括インストール
python3 -m pip install --break-system-packages -U pip \
  mlflow polars[all] lightgbm catboost xgboost \
  optuna[lightgbm] duckdb
```

```
python3 - <<'PY'
import importlib, pkg_resources, textwrap

pkgs = [
    "mlflow", "polars", "lightgbm", "catboost", "xgboost",
    "optuna", "duckdb"
]
for p in pkgs:
    try:
        v = pkg_resources.get_distribution(p).version
        print(f"{p:10}  {v}")
    except Exception as e:
        print(f"{p:10}  NOT FOUND ({e.__class__.__name__})")
PY
```

---

## 3. ワークフロー

```bash
# 0. 初回のみ
pip install -r requirements.txt  # 上記ツール一覧

# 1. 単一 fold 学習
make train F=0 S=42              # → mlruns/ にログ

# 2. 5-fold 並列学習
make kfold N=5 S=42

# 3. ハイパラ探索（任意）
make tuner TRIALS=50

# 4. 推論 → 提出
make predict RUN_IDS=<id1,id2,...>
make submit C=<comp-name> S=42
```

**最低限の Makefile**

```make
train:
	mlflow run . -P fold=$(F) -P seed=$(S)

kfold:
	for i in $(shell seq 0 $(N-1)); do \
	  make train F=$$i S=$(S); \
	done

predict:
	python predict.py $(RUN_IDS)

submit:
	kaggle competitions submit -c $(C) -f submission.csv -m "seed$(S)"
```

---

## 4. アンサンブル & スタッキング

| レベル         | 手法                         | 所要時間   | 実装ポイント                              |
| ----------- | -------------------------- | ------ | ----------------------------------- |
| ① 平均/順位ブレンド | 加重平均 or rank 平均            | 10 分   | `polars` で 3 ファイル読み込み → 重み付け保存      |
| ② 重み最適化     | Optuna で RMSE/AUC 最小化      | 30 分   | `study.optimize()` だけ。変数 2〜3 個なので高速 |
| ③ スタッキング    | OOF 予測 → LGBM/LogReg メタモデル | 1–2 時間 | fold を揃える & OOF だけで学習することが鍵         |

> **Tips**
>
> * 相関を下げるために CatBoost は違う seed / GPU 設定で回すと効果大
> * メタモデルは `max_depth<=2` に制限し過学習を防ぐ

---

## 5. よくある質問 (FAQ)

| 質問                  | 回答                                                                    |
| ------------------- | --------------------------------------------------------------------- |
| 依存パッケージが競合したら？      | Kaggle しか触らないなら無視。どうしても困れば `python -m venv .venv` で局所隔離               |
| GPU が無い場合は？         | CatBoost/XGBoost を CPU 版に。LightGBM の `num_threads` を最適化すれば銅圏内まで行ける例多数 |
| Notebook で可視化したくなった | `mlflow ui` + ブラウザで十分。どうしてもセル実行が必要な時だけ Jupyter を後付けインストール             |
| 順位が伸び悩む             | ① 特徴量追加 → ② ハイパラ再調整 → ③ アンサンブル順に試すと効率的                                |

---

## 6. まとめ

```text
WSL (Ubuntu 標準 Python)
 ├── MLflow
 ├── polars
 ├── LightGBM / CatBoost / XGBoost
 ├── (Optuna)
 └── Bash / Makefile
```

*環境構築ゼロ* → *モデル改善に全力*。この構成でまず 1 コンペ完走し、
詰まった箇所だけ局所的にツール追加すれば OK。さらに質問があれば気軽にどうぞ！


