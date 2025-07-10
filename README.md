# Solid ML Stack

高速でスケーラブルなKaggle特化機械学習パイプライン

## 特徴

### 🎯 Kaggle最適化
- **コンペ提出までのワークフロー最適化**: 前処理→特徴量エンジニアリング→モデル学習→アンサンブル→提出の一連のフローを自動化
- **再利用可能な設計**: 関数・クラスベースで構成し、異なるコンペでも簡単に再利用可能
- **CPU特化**: XGBoost/LightGBM/CatBoostなどツリーベースモデルを中心とした高速学習

### 🔧 モジュール化設計
- **前処理**: 欠損値処理、外れ値除去、スケーリング、エンコーディング
- **特徴量エンジニアリング**: 数値変換、カテゴリエンコーディング、交互作用特徴量、時系列特徴量
- **モデル学習**: XGBoost、LightGBM、CatBoost、線形モデル
- **パラメータ探索**: Grid Search、Random Search、Bayesian Optimization、Optuna
- **アンサンブル**: 平均化、重み付き平均、スタッキング、Voting

### 📊 タブラーデータ特化
- CSVなどの表形式データに特化
- pandas→scikit-learnパイプラインベース
- 画像・テキスト・時系列Deep Learningは対象外

## インストール

```bash
# 基本依存関係
pip install -e .

# 最適化ライブラリ（オプション）
pip install -e .[optimization]

# 可視化ライブラリ（オプション）
pip install -e .[visualization]

# 開発用ツール
pip install -e .[dev]
```

## クイックスタート

### 基本的な使い方

```python
import pandas as pd
from preprocessing import Preprocessor
from features.engineering import AutoFeatureEngineer
from modeling import ModelFactory
from submission import SubmissionGenerator

# データ読み込み
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 前処理
preprocessor = Preprocessor()
X_train, X_val, y_train, y_val = preprocessor.prepare_data(train_df, 'target')
X_train_processed, X_val_processed = preprocessor.process_train_test(X_train, X_val, y_train)

# 特徴量エンジニアリング
feature_engineer = AutoFeatureEngineer()
X_train_features = feature_engineer.fit_transform(X_train_processed, y_train)
X_val_features = feature_engineer.pipeline.transform(X_val_processed)

# モデル学習
factory = ModelFactory()
models = factory.get_default_models(target_type='regression')

trained_models = {}
for model in models:
    model.fit(X_train_features, y_train, X_val_features, y_val)
    trained_models[model.config.name] = model

# 予測・提出ファイル生成
X_test_processed = preprocessor.transform(test_df.drop(columns=['id']))
X_test_features = feature_engineer.pipeline.transform(X_test_processed)

predictions = {name: model.predict(X_test_features) 
               for name, model in trained_models.items()}

submission_gen = SubmissionGenerator()
submission_path = submission_gen.create_ensemble_submission(
    predictions, test_df['id'].values, 'target', 'id'
)
```

### コマンドライン実行

```bash
# フルワークフロー実行
python scripts/kaggle_workflow.py \
    --train-path data/raw/train.csv \
    --test-path data/raw/test.csv \
    --target-col target \
    --problem-type regression \
    --optimize \
    --ensemble

# Makefileを使った実行
make kaggle-regression TRAIN=data/raw/train.csv TEST=data/raw/test.csv TARGET=target
make kaggle-classification TRAIN=data/raw/train.csv TEST=data/raw/test.csv TARGET=class
```

## プロジェクト構造

```
src/
├── preprocessing/          # 前処理モジュール
│   ├── preprocessor.py    # メイン前処理クラス
│   ├── pipeline.py        # 前処理パイプライン
│   └── transformers.py    # 個別変換器
├── features/              # 特徴量エンジニアリング
│   └── engineering/       # 特徴量生成
│       ├── base.py        # ベース特徴量生成器
│       ├── numeric.py     # 数値特徴量
│       ├── categorical.py # カテゴリ特徴量
│       ├── interaction.py # 交互作用特徴量
│       ├── datetime.py    # 日時特徴量
│       ├── aggregation.py # 集約特徴量
│       └── pipeline.py    # 特徴量パイプライン
├── modeling/              # モデル学習
│   ├── base.py           # ベースモデル
│   ├── tree_models.py    # ツリーモデル（XGBoost、LightGBM、CatBoost）
│   ├── linear_models.py  # 線形モデル
│   ├── ensemble.py       # アンサンブル手法
│   └── factory.py        # モデルファクトリ
├── optimization/          # パラメータ探索
│   ├── base.py           # ベース最適化クラス
│   ├── grid_search.py    # グリッドサーチ
│   ├── random_search.py  # ランダムサーチ
│   ├── bayesian_optimization.py # ベイジアン最適化
│   ├── optuna_optimizer.py # Optuna最適化
│   └── factory.py        # 最適化ファクトリ
├── evaluation/           # モデル評価
│   └── metrics.py        # 評価指標
├── submission/           # 提出ファイル生成
│   └── submission_generator.py
└── utils/                # ユーティリティ
    ├── base.py           # ベースユーティリティ
    ├── config.py         # 設定管理
    └── io.py             # ファイル入出力
```

## 使用例

### 1. カスタム前処理パイプライン

```python
from preprocessing import PreprocessingPipeline
from preprocessing.transformers import *

# カスタム前処理パイプライン
pipeline = PreprocessingPipeline()
pipeline.add_step('missing', MissingValueHandler(numeric_strategy='median'))
pipeline.add_step('outliers', OutlierHandler(method='zscore', threshold=3))
pipeline.add_step('encoding', CategoricalEncoder(method='target'))
pipeline.add_step('scaling', NumericScaler(method='robust'))

X_processed = pipeline.fit_transform(X_train, y_train)
```

### 2. 特徴量エンジニアリング

```python
from features.engineering import FeatureEngineeringPipeline

# 特徴量パイプライン構築
fe_pipeline = FeatureEngineeringPipeline()
fe_pipeline.add_numeric_features(['log', 'sqrt', 'square'])
fe_pipeline.add_categorical_features(min_frequency=5)
fe_pipeline.add_interaction_features(max_interactions=100)
fe_pipeline.add_polynomial_features(degree=2)

X_features = fe_pipeline.fit_transform(X_train, y_train)
```

### 3. パラメータ最適化

```python
from optimization import OptunaOptimizer, OptimizationConfig

# Optuna最適化
search_space = {
    'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
    'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3}
}

config = OptimizationConfig(search_space=search_space, n_trials=100)
optimizer = OptunaOptimizer(config)

result = optimizer.optimize(model, X_train, y_train, X_val, y_val)
best_model = result.apply_best_params()
```

### 4. アンサンブル

```python
from modeling import StackingEnsemble, create_kaggle_models

# 多様なモデル作成
models = create_kaggle_models(target_type='regression')

# スタッキングアンサンブル
stacking_model = StackingEnsemble(base_models=models, cv_folds=5)
stacking_model.fit(X_train, y_train, X_val, y_val)

predictions = stacking_model.predict(X_test)
```

## コンフィグ設定

```python
from config.kaggle_config import KaggleConfig, ConfigPresets

# 回帰問題用設定
config = ConfigPresets.regression_competition()

# 分類問題用設定
config = ConfigPresets.classification_competition()

# カスタム設定
config = KaggleConfig(
    problem_type='regression',
    preprocessing={
        'handle_missing': True,
        'handle_outliers': True,
        'outlier_threshold': 2.0
    },
    feature_engineering={
        'numeric_features': True,
        'polynomial_features': True,
        'max_interactions': 150
    }
)
```

## テスト実行

```bash
# 全テスト実行
make test

# 高速テスト（slowマーカーを除外）
make test-fast

# ユニットテストのみ
make test-unit

# 統合テストのみ
make test-integration

# カバレッジ付きテスト
make test-coverage

# スモークテスト
make test-smoke
```

## 開発ガイドライン

### コード品質
- **型ヒント**: 全ての関数・メソッドに型アノテーション
- **docstring**: 主要クラス・関数にGoogleスタイルドキュメント
- **テスト**: pytestによる単体テスト
- **フォーマット**: blackによる自動フォーマット

### パフォーマンス
- **CPU最適化**: GPU不要でローカル環境で高速実行
- **メモリ効率**: 大容量データでもメモリ効率的に処理
- **並列処理**: 可能な箇所での並列化実装

### セキュリティ
- **秘匿情報**: API キーや認証情報のハードコード禁止
- **入力検証**: 外部データの適切な検証・サニタイズ

## ライセンス

MIT License

## 貢献

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## サポート

- Issues: GitHub Issues で質問・バグ報告
- Discussions: GitHub Discussions で一般的な議論