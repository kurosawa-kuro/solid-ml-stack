# solid-ml-stack



## 推奨ミニマム・スタック

| レイヤ               | ツール                                          | 役割                                                                       | 使い方のポイント                                                                                                     |
| ----------------- | -------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| **実験管理**          | **MLflow (CLI / Python API)**                | すべての実験を `python train.py ...` で回し<br>自動でメトリクス・パラメータ・モデルをログ。<br>UI で比較も可。 | - `mlflow run . -P fold=0 -P seed=42` のようにパラメータ化すると **Notebook なしで一気に全fold実行**<br>- 追い込み時は `mlflow ui` で差分分析 |
| **ETL & 前処理**     | **polars**                                   | 超高速 CSV → DataFrame。読み込み・結合・集計はほぼ1行。                                     | - `scan_csv()`→lazy変換→`collect()` で並列処理<br>- LightGBM 用に `to_pandas()` で手軽に受け渡し                              |
| **モデル**           | scikit-learn / LightGBM / CatBoost / XGBoost | 定番ライブラリ中心で十分銅メダル圏内。                                                      | - `mlflow.sklearn.autolog()` などで自動ログ<br>- 大規模特徴量なら CatBoost GPU も検討                                          |
| **ダッシュボード**       | **Metabase**                                 | コンペ期間中のスコア推移・特徴量重要度を<br>可視化して作戦会議。                                       | - MLflow の `runs.meta.yaml` や自作 CSV を<br>**DuckDB に取り込み**→ Metabase で接続<br>- SQL で「fold 別/feature 単位」の傾向を即確認 |
| **データストア (ローカル)** | DuckDB                                       | Metabase がつなげる軽量 DB。<br>polars とも相性抜群。                                   | - `con.execute("COPY my_table TO 'results.parquet'")`<br>でスナップショット管理も楽                                       |
| **実行基盤**          | Makefile or Bash スクリプト                       | `make train`, `make submit` などで<br>一撃フルパイプライン。                           | - 例: `make all` → データDL → 前処理 → 学習 → 予測 → 提出<br>- GitHub Actions (CI) は不要、ローカル一発で十分                          |

---

## 典型ワークフロー（サンプル）

```bash
# 0. 一度だけ
pip install mlflow polars lightgbm kaggle duckdb metabase-driver-duckdb

# 1. 学習（fold=0, seed=42）
mlflow run . -P fold=0 -P seed=42

# 2. 5-fold まとめて回す
seq 0 4 | xargs -I{} -P5 mlflow run . -P fold={} -P seed=42

# 3. 予測 & アンサンブル
python predict.py --run-ids $(mlflow runs list -q "tags.seed='42'" -o id)

# 4. 提出
kaggle competitions submit -c <comp-name> -f submission.csv -m "seed42 5fold"
```

* **train.py**:

  1. polars で CSV を **lazy 読み込み → 前処理**
  2. pandas に変換して LightGBM fit
  3. MLflow でメトリクス／モデル／feature importance をログ
  4. 予測ファイルは `artifacts/pred_fold{n}.csv` に保存

* **predict.py**:
  MLflow から run\_id 指定でモデル取得 → 全 fold の予測を平均 → submission.csv 生成

---

## Metabase 接続イメージ

1. `duckdb results.duckdb`

   ```sql
   CREATE TABLE runs AS
   SELECT * FROM read_csv_auto('mlruns/**/metrics.csv');  -- MLflow 各 run のメトリクス
   CREATE TABLE feature_imp AS
   SELECT * FROM read_parquet('mlruns/**/feature_importance.parquet');
   ```
2. Metabase で **DuckDB ドライバ**を選択 → `results.duckdb` を指定
3. ダッシュボード例

   * スコア vs 学習時間 散布図
   * fold 別 AUC 推移
   * 上位20特徴量の重要度バー

---

## なぜ Notebook を捨てて OK か

| Notebook で困る点      | 代替策 (今回のスタック)                |                                  |
| ------------------ | ---------------------------- | -------------------------------- |
| セル実行順序がズレて再現が難しい   | **MLflow + スクリプト** で常にクリーン実行 |                                  |
| 複数パラメータのグリッドサーチが面倒 | \`seq                        | xargs mlflow run\` で<br>並列かけ捨て運用 |
| 大量ログが散逸            | MLflow が自動で一元管理              |                                  |

---

### まとめ

* **MLflow** = 実験ログ & モデル管理、**Metabase** = 戦略ダッシュボード。
* Notebook は捨て、**polars + Python スクリプト**で“回し切る”。
* DuckDB をハブにすれば Metabase 連携も爆速。

これで「好きなツールだけ残しつつ、一気に回せる」銅メダル特化環境になります。試してみて、ハマりどころがあればまた相談してください！
