以下では、いただいた **solid‑ml‑stack** を「Kaggle銅メダル獲得だけ」を目的に**さらに研ぎ澄まし**、
――余計な機能・概念を削ぎ落としつつ“勝つための尖った設計”――にリファクタリングしました。

---

## 1. コンセプト整理

| 目的                                  | 必要機能                  | 採用コンポーネント                               | なぜそれだけで十分か                                              |
| ----------------------------------- | --------------------- | --------------------------------------- | ------------------------------------------------------- |
| **試行回数を最大化**                        | 再現性のある一括実行            | **Makefile / Bash**<br>＋ **MLflow CLI** | *Docker, CI/CD, Notebook* を排し、CPU/GPU を**研究に 100 % 投入** |
| **特徴量とモデルを素早く切り替え**                 | 高速 ETL／結合／集計          | **polars** (lazy)                       | pandas の 3‑10 倍速。学習スクリプト側は pandas 互換 API で変更ゼロ          |
| **どこで点数が伸びたか即把握**                   | 実験ログ可視化               | **MLflow UI**                           | Weight & Biases 等は不要。学習中に `localhost:5000` を開くだけ        |
| **意思決定をチームに共有**<br>（個人でも“過去の自分”と議論） | 集計結果の定点観測             | **DuckDB + Metabase**                   | SQLite 感覚で使える列指向 DB。GUI は**後回し可**：SQL だけでも完結            |
| **汎用 Tabular モデル**                  | 高速／高精度 GBDT           | **LightGBM / CatBoost / XGBoost**       | 90 % 以上のコンペで**この3つで銅圏内**。深層学習は持ち込まない                    |
| **ハイパラ最適化**                         | Bayesian Optimization | **Optuna (オプション)**                      | MLflow との自動連携が3行。必要になった時だけ追加                            |

> 🔪 **“やらないこと”を先に決める**
>
> * Docker / Poetry / Conda : **WSL グローバル pip** に一本化
> * Notebook : 完全排除
> * クラウド CI : ローカル GPU が最速

---

## 2. 改訂ミニマム・スタック

| レイヤ           | ツール                           | 使いこなしの要点                                        | インストール（初回のみ）                               |
| ------------- | ----------------------------- | ----------------------------------------------- | ------------------------------------------ |
| **環境管理**      | `pyenv` + `pipx`              | グローバル汚染を最小化。`pipx run mlflow` も可                | `curl https://pyenv.run \| bash`           |
| **実験管理**      | **MLflow 2.x**                | `mlflow run -P cfg=baseline.yaml` で1行実行         | `pip install mlflow==2.*`                  |
| **構成管理**      | **Hydra Lite**                | config ファイルを1枚だけ管理（fold, seed, Features On/Off） | `pip install hydra-core --no-binary :all:` |
| **ETL**       | **polars**                    | `scan_csv().with_columns([...]).collect()`      | `pip install polars[all]`                  |
| **学習**        | LightGBM / CatBoost / XGBoost | GPU ↔ CPU を config で切替                          | `pip install lightgbm catboost xgboost`    |
| **ハイパラ (任意)** | Optuna                        | `study.optimize(objective, n_trials=50)`        | `pip install optuna[lightgbm]`             |
| **DB**        | DuckDB                        | `con.execute("COPY ...")` で履歴永続化                | `pip install duckdb`                       |
| **可視化 (任意)**  | Metabase                      | 実験後に docker-compose で一時起動しても可                   | *必要時のみ*                                    |

> 💡 \*\*Metabase も「後付けオプション」\*\*として扱い、コンペ前半はログと SQL だけで回すと軽い。

---

## 3. 改訂ワークフロー

```bash
# ──────────── 0. 初回セットアップ
pyenv install 3.12.2 && pyenv global 3.12.2
pip install -r requirements.txt                     # 上表のパッケージ集合

# ──────────── 1. 単一 fold 学習
make train FOLD=0 SEED=42 CFG=baseline.yaml         # → mlruns/ にログ

# ──────────── 2. k-fold 並列学習
make kfold N=5 SEED=42                              # GNU parallel 利用

# ──────────── 3. ハイパラ探索 (任意)
make tuner TRIALS=50                                # Optuna ←→ MLflow 連携

# ──────────── 4. 推論 & 提出
make predict                                         # submission.csv 生成
make submit COMP=<competition-name>
```

**Makefile 抜粋**

```make
train:
	mlflow run . -P fold=$(FOLD) -P seed=$(SEED) -P cfg=$(CFG)

kfold:
	seq 0 $(shell echo $(N)-1 | bc) | \
	xargs -P$(N) -I{} make train FOLD={} SEED=$(SEED) CFG=$(CFG)

tuner:
	python tune.py --trials $(TRIALS)

predict:
	python predict.py --seed $(SEED)

submit:
	kaggle competitions submit -c $(COMP) -f submission.csv -m "seed$(SEED) $(N)-fold"
```

---

## 4. スクリプト雛形（要点だけ）

### `train.py`

1. **polars** で `scan_csv()` → 前処理 (`lazy` で列演算)
2. 必要なら `.to_pandas()` で LightGBM に適合
3. **MLflow autolog** ＋ **Hydra config** で完全再現
4. Fold ごとの予測を `artifacts/` に保存

### `predict.py`

```python
import mlflow
import pandas as pd
import glob, numpy as np, sys

run_ids = sys.argv[1:]                # make predict で受け取る
preds = [mlflow.pyfunc.load_model(f"runs:/{r}/model").predict(... ) for r in run_ids]
pd.DataFrame(np.mean(preds, axis=0), columns=["target"]).to_csv("submission.csv", index=False)
```

---

## 5. “一段上” を狙う小技集

| シーン                 | 小技                                                            | 実装コスト |
| ------------------- | ------------------------------------------------------------- | ----- |
| **前処理が重くなった**       | polars `Streaming CSV` ＋ `SQLContext` でメモリ削減                  | ★☆☆   |
| **特徴量が爆増**          | 特徴量毎に **shapley value** を計算し MLflow Artifact に保存              | ★★☆   |
| **stack/blend の調整** | `kaggle API leaderboard_download` → 自己 LB で **線形回帰 blending** | ★★☆   |
| **GPU 環境がある**       | CatBoost GPU, XGBoost GPU版を入れ替え (`pip install catboost`)      | ★☆☆   |
| **順位頭打ち**           | Optuna で *bagging\_temperature*, *feature\_fraction* を自動探索    | ★★☆   |

---

## 6. まとめ ― 結局こうなる

```text
WSL (pyenv)
 ├── MLflow ──┐
 │            │── DuckDB ←→ (Metabase)  ※必要時だけ
 ├── polars   │
 └── GBDT系   │
              └── Bash / Makefile  (Hydra config)
```

* **すべて CLI & スクリプトのみ**。GUI は確認用 UI だけ。
* 「**学習・予測・提出**」を Makefile に閉じ込め、**思考コスト＝モデル改善**へ集中。
* 追加ツールは Optuna と Metabase の *2 枚* だけ。必要になった瞬間に足す。

> 👑 **目的は銅メダル。余計な環境構築に 1 秒も使わない。**
> まずはこの形でコンペを 1 つ完走し、ボトルネックが出た所だけ局所的に強化すれば OK です。

ご不明点や「ここがまだ重い／面倒」などあれば、遠慮なくお知らせください！
