# ML Module Refactoring

## 📋 概要

MLモジュールのリファクタリングにより、以下の改善を実現しました：

- **型安全性の向上**: TypeScriptライクな型ヒント
- **設定の一元管理**: 集中化された設定システム
- **エラーハンドリング**: 包括的な例外処理
- **テスト可能性**: モジュラー設計によるテストしやすさ
- **拡張性**: 新しいモデルの追加が容易

## 🏗️ アーキテクチャ

### 新しい構造

```
src/ml/
├── base_model.py          # 抽象基底クラス
├── config.py              # 設定管理
├── model_factory.py       # モデルファクトリー
├── utils.py               # ユーティリティ関数
├── base_improved.py       # 改善されたデータ処理
├── train_refactored.py    # リファクタリングされた学習スクリプト
├── test_models.py         # モデルテストスクリプト
├── example_usage.py       # 使用例スクリプト
├── models/                # モデル実装
│   ├── xgb_model.py       # XGBoostモデル
│   ├── cat_model.py       # CatBoostモデル
│   ├── lgbm_model.py      # LightGBMモデル
│   └── ensemble_model.py  # エンサンブル・スタッキングモデル
└── README.md              # このファイル
```

### 主要コンポーネント

#### 1. BaseModel (抽象基底クラス)
```python
class BaseModel(ABC):
    def train(self, X, y) -> Any
    def predict(self, X) -> np.ndarray
    def fit_predict(self, X, y) -> ModelResult
```

#### 2. ModelConfig (設定管理)
```python
@dataclass
class MLConfig:
    seed: int = 42
    test_size: float = 0.2
    xgb_config: XGBoostConfig
    cat_config: CatBoostConfig
    lgbm_config: LightGBMConfig
    ensemble_config: EnsembleConfig
```

#### 3. ModelFactory (ファクトリー)
```python
class ModelFactory:
    def create_model(self, model_name: str) -> BaseModel
    def get_available_models(self) -> list
```

## 🚀 使用方法

### 基本的な使用例

```python
from ml.config import DEFAULT_CONFIG
from ml.model_factory import model_factory
from ml.base_improved import DataLoader

# データ読み込み
loader = DataLoader()
X, y = loader.load_data("path/to/db.duckdb")

# モデル作成と学習
model = model_factory.create_model("xgb")
result = model.fit_predict(X, y)

print(f"RMSE: {result.score:.4f}")
```

### 設定のカスタマイズ

```python
from ml.config import MLConfig, XGBoostConfig

# カスタム設定
config = MLConfig(
    seed=123,
    xgb_config=XGBoostConfig(
        num_boost_round=200,
        max_depth=8,
        learning_rate=0.05
    )
)

# ファクトリーに設定を適用
from ml.model_factory import ModelFactory
factory = ModelFactory(config)
model = factory.create_model("xgb")
```

### 全モデルの学習

```python
from ml.train_refactored import MLTrainer

trainer = MLTrainer()
results = trainer.train_all_models(X, y)

# 最良モデルを取得
best_model = trainer.get_best_model()
print(f"Best model: {best_model}")
```

## 🔧 設定オプション

### XGBoost設定
- `objective`: 目的関数 ("reg:squarederror")
- `num_boost_round`: ブースティング回数 (100)
- `max_depth`: 最大深さ (6)
- `learning_rate`: 学習率 (0.1)
- `subsample`: サブサンプリング率 (0.8)
- `colsample_bytree`: 特徴量サブサンプリング率 (0.8)

### CatBoost設定
- `iterations`: 反復回数 (100)
- `depth`: 深さ (6)
- `learning_rate`: 学習率 (0.1)
- `loss_function`: 損失関数 ("RMSE")
- `verbose`: 詳細出力 (False)

### LightGBM設定
- `objective`: 目的関数 ("regression")
- `num_boost_round`: ブースティング回数 (100)
- `max_depth`: 最大深さ (6)
- `learning_rate`: 学習率 (0.1)
- `subsample`: サブサンプリング率 (0.8)
- `colsample_bytree`: 特徴量サブサンプリング率 (0.8)
- `verbosity`: 詳細レベル (-1)

### エンサンブル設定
- `weights`: 重み辞書 (None)
- `method`: エンサンブル方法 ("average", "weighted", "median")

## 📊 結果管理

### 結果の保存
```python
from ml.utils import save_result, save_predictions

# スコアを保存
save_result("xgb", 0.1234)

# 予測結果を保存
save_predictions("xgb", y_pred, timestamp, "models/")
```

### 結果の読み込み
```python
from ml.utils import load_latest_results, get_best_model

# 最新結果を取得
results = load_latest_results()

# 最良モデルを取得
best = get_best_model()
```

## 🧪 テスト

### テストスクリプトの実行
```bash
cd src/ml
python test_models.py
```

### 使用例の実行
```bash
cd src/ml
python example_usage.py
```

### 単体テストの例
```python
import pytest
from ml.models.xgb_model import XGBoostModel
from ml.config import ModelConfig

def test_xgb_model():
    config = ModelConfig(seed=42)
    model = XGBoostModel(config)
    
    # テストデータ
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    # 学習と予測
    result = model.fit_predict(X, y)
    
    assert result.score >= 0
    assert len(result.y_pred) == len(y)
```

## 🔄 移行ガイド

### 既存コードからの移行

#### Before (旧コード)
```python
from xgb_runner import fit_predict_xgb
_, y_pred, score = fit_predict_xgb(X, y)
```

#### After (新コード)
```python
from ml.model_factory import model_factory
model = model_factory.create_model("xgb")
result = model.fit_predict(X, y)
score = result.score
y_pred = result.y_pred
```

## 🆕 新機能

### 特徴量重要度
```python
# XGBoost
xgb_model = model_factory.create_model("xgb")
result = xgb_model.fit_predict(X, y)
importance = xgb_model.get_feature_importance()

# CatBoost
cat_model = model_factory.create_model("cat")
result = cat_model.fit_predict(X, y)
importance = cat_model.get_feature_importance()

# LightGBM
lgbm_model = model_factory.create_model("lgbm")
result = lgbm_model.fit_predict(X, y)
importance = lgbm_model.get_feature_importance_gain()  # ゲインベース
```

### エンサンブルモデル
```python
# 平均エンサンブル
ensemble_model = model_factory.create_model("ensemble")
result = ensemble_model.fit_predict(X, y)

# スタッキング
stacking_model = model_factory.create_model("stacking")
result = stacking_model.fit_predict(X, y)
```

### カスタムエンサンブル
```python
from ml.config import EnsembleConfig

# 重み付きエンサンブル
config = EnsembleConfig(
    weights={"xgb": 0.4, "cat": 0.3, "lgbm": 0.3},
    method="weighted"
)

ensemble_model = model_factory.create_model("ensemble", ensemble_config=config)
result = ensemble_model.fit_predict(X, y)
```

## 📈 パフォーマンス改善

1. **並列処理**: 複数モデルの並列学習
2. **キャッシュ**: 特徴量エンジニアリングの結果キャッシュ
3. **メモリ最適化**: 大規模データセットの効率的処理

## 🔮 今後の拡張

1. **新しいモデル**: ニューラルネットワーク、SVM等
2. **ハイパーパラメータ最適化**: Optuna統合
3. **実験管理**: MLflow統合
4. **モデル解釈**: SHAP、LIME統合
5. **自動ML**: AutoML機能

## 📝 注意事項

- 既存の`train.py`は後方互換性のために残されています
- 新しいコードは`train_refactored.py`を使用してください
- 設定ファイルは`config.py`で一元管理されています
- ログ出力は`logging`モジュールを使用しています
- 各モデルライブラリ（XGBoost、CatBoost、LightGBM）のインストールが必要です

## 🛠️ インストール要件

```bash
pip install xgboost catboost lightgbm scikit-learn pandas numpy
```

## 📚 参考資料

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/) 