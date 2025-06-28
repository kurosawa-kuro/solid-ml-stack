#!/usr/bin/env python3
"""ML実験結果レポート生成スクリプト"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json

# srcディレクトリをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config import Config
from ml.models.model_factory import model_factory


def load_ml_results(config: Config) -> pd.DataFrame:
    """ML実験結果を読み込み"""
    metrics_file = config.project_root / "artifacts" / "metrics" / "metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        # JSONデータをDataFrameに変換
        rows = []
        for model_name, runs in metrics_data.items():
            for run in runs:
                rows.append({
                    'model': model_name,
                    'rmse': run.get('rmse', 0),
                    'mae': run.get('mae', None),
                    'r2': run.get('r2', None),
                    'timestamp': run.get('timestamp', '')
                })
        
        return pd.DataFrame(rows)
    else:
        print(f"Warning: {metrics_file} not found")
        return pd.DataFrame()


def generate_model_comparison_report(config: Config) -> str:
    """モデル比較レポートを生成"""
    results_df = load_ml_results(config)
    
    if results_df.empty:
        return "No ML results found. Run 'make ml-all' first."
    
    report = []
    report.append("=" * 60)
    report.append("ML Model Performance Comparison Report")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 基本統計
    report.append("📊 Performance Summary:")
    report.append("-" * 30)
    
    if 'model' in results_df.columns and 'rmse' in results_df.columns:
        best_model = results_df.loc[results_df['rmse'].idxmin()]
        report.append(f"🏆 Best Model: {best_model['model']} (RMSE: {best_model['rmse']:.2f})")
        report.append("")
        
        # モデル別性能
        report.append("📈 Model Performance Ranking (by RMSE):")
        sorted_results = results_df.sort_values('rmse')
        for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
            report.append(f"  {i}. {row['model']:12} - RMSE: {row['rmse']:8.2f}, R²: {row.get('r2', 'N/A'):6.3f}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def generate_feature_importance_report(config: Config) -> str:
    """特徴量重要度レポートを生成"""
    report = []
    report.append("🔍 Feature Importance Analysis")
    report.append("=" * 40)
    
    # 利用可能なモデルで特徴量重要度を取得
    available_models = ['xgb', 'cat', 'lgbm']
    
    for model_name in available_models:
        try:
            model = model_factory.create_model(model_name)
            # ダミーデータで特徴量重要度を取得（実際のデータがない場合）
            report.append(f"\n📊 {model_name.upper()} Feature Importance:")
            report.append("-" * 30)
            
            # 実際の特徴量重要度は学習後に取得する必要があるため、
            # ここでは説明のみ
            report.append("  (Feature importance available after model training)")
            
        except Exception as e:
            report.append(f"  Error loading {model_name}: {e}")
    
    return "\n".join(report)


def generate_data_pipeline_report(config: Config) -> str:
    """データパイプライン状況レポートを生成"""
    report = []
    report.append("🔄 Data Pipeline Status")
    report.append("=" * 30)
    
    # データベース接続確認
    try:
        import duckdb
        with duckdb.connect(str(config.db_path)) as conn:
            tables = conn.execute("SHOW TABLES").fetchall()
            
            report.append(f"📁 Database: {config.db_path}")
            report.append(f"📊 Tables: {len(tables)}")
            
            for table in tables:
                table_name = table[0]
                count_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                count = count_result[0] if count_result else 0
                report.append(f"  - {table_name}: {count:,} rows")
                
    except Exception as e:
        report.append(f"❌ Database error: {e}")
    
    # ファイル存在確認
    report.append(f"\n📂 Raw data: {'✅' if config.raw_data_path.exists() else '❌'} {config.raw_data_path}")
    report.append(f"📂 Artifacts: {'✅' if config.artifacts_dir.exists() else '❌'} {config.artifacts_dir}")
    
    return "\n".join(report)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Generate ML experiment reports")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--sections", "-s", nargs="+", 
                       choices=["comparison", "features", "pipeline", "all"],
                       default=["all"], help="Report sections to include")
    
    args = parser.parse_args()
    
    config = Config()
    
    # レポート生成
    report_sections = []
    
    if "all" in args.sections or "comparison" in args.sections:
        report_sections.append(generate_model_comparison_report(config))
    
    if "all" in args.sections or "features" in args.sections:
        report_sections.append(generate_feature_importance_report(config))
    
    if "all" in args.sections or "pipeline" in args.sections:
        report_sections.append(generate_data_pipeline_report(config))
    
    # レポート出力
    full_report = "\n\n".join(report_sections)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"Report saved to: {output_path}")
    else:
        print(full_report)


if __name__ == "__main__":
    main() 