import pandas as pd
import numpy as np
import os
import json
import sys
import time
from config import PROCESSED_DIR, TOP_N_AUTO_FEATURES


def load_data_and_cols():
    cols_path = os.path.join(PROCESSED_DIR, "columns.json")
    with open(cols_path, "r") as f:
        cols = json.load(f)

    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

    ts_col = cols["timestamp"]
    for df in [train, val, test]:
        df[ts_col] = pd.to_datetime(df[ts_col])

    return train, val, test, cols


def run_baseline(train, val, test, cols):
    from manual_features import build_features_by_periods
    from train_and_evaluate import train_catboost, evaluate_model

    print("\n" + "=" * 60)
    print("ЭТАП 1: BASELINE — РУЧНЫЕ ПРИЗНАКИ + CATBOOST")
    print("=" * 60)

    full_history = train.copy()

    print("\nСтрою признаки для валидации...")
    val_features, val_feat_cols = build_features_by_periods(val, full_history, cols)

    print(f"Получено {len(val_feat_cols)} ручных признаков")
    print(f"Признаки: {val_feat_cols}")

    train_half = int(len(train) * 0.7)
    train_part1 = train.iloc[:train_half]
    train_part2 = train.iloc[train_half:]

    print("\nСтрою признаки для обучения...")
    train_features, train_feat_cols = build_features_by_periods(
        train_part2, train_part1, cols
    )

    common_cols = list(set(train_feat_cols) & set(val_feat_cols))
    print(f"Общих признаков: {len(common_cols)}")

    if len(common_cols) == 0:
        print("ОШИБКА: нет общих признаков!")
        return None, None

    print("\nОбучаю CatBoost baseline...")
    model = train_catboost(train_features, val_features, common_cols, "baseline")

    print("\nОцениваю на валидации...")
    val_metrics = evaluate_model(model, val_features, common_cols, cols, "baseline_val")

    print("\nСтрою признаки для теста...")
    full_before_test = pd.concat([train, val], ignore_index=True)
    test_features, test_feat_cols = build_features_by_periods(
        test, full_before_test, cols
    )

    test_common = list(set(common_cols) & set(test_feat_cols))
    print(f"\nОцениваю на тесте...")
    test_metrics = evaluate_model(model, test_features, test_common, cols, "baseline_test")

    return model, common_cols


def run_auto_features(train, val, test, cols):
    from auto_features import build_auto_features, select_features_by_importance
    from train_and_evaluate import train_catboost, evaluate_model

    print("\n" + "=" * 60)
    print("ЭТАП 2: АВТОМАТИЧЕСКИЕ ПРИЗНАКИ + CATBOOST")
    print("=" * 60)

    full_history = train.copy()

    print("\nГенерирую и считаю автоматические признаки для валидации...")
    start_time = time.time()
    val_auto, val_auto_cols, specs = build_auto_features(val, full_history, cols)
    auto_time = time.time() - start_time
    print(f"Время генерации: {auto_time:.1f} сек")
    print(f"Сгенерировано {len(val_auto_cols)} автоматических признаков")

    numeric_auto_cols = [c for c in val_auto_cols
                         if val_auto[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    if len(numeric_auto_cols) == 0:
        print("Нет числовых автоматических признаков!")
        return None, None

    print("\nОтбор лучших признаков...")
    selected_cols = select_features_by_importance(
        val_auto, val_auto["label"], numeric_auto_cols, TOP_N_AUTO_FEATURES
    )
    print(f"Отобрано {len(selected_cols)} признаков")

    train_half = int(len(train) * 0.7)
    train_part1 = train.iloc[:train_half]
    train_part2 = train.iloc[train_half:]

    print("\nСчитаю автоматические признаки для обучения...")
    train_auto, train_auto_cols, _ = build_auto_features(
        train_part2, train_part1, cols, specs
    )

    common_auto = list(set(selected_cols) & set(train_auto_cols))
    print(f"Общих отобранных признаков: {len(common_auto)}")

    if len(common_auto) == 0:
        print("ОШИБКА: нет общих авто-признаков!")
        return None, None

    print("\nОбучаю CatBoost на автоматических признаках...")
    model_auto = train_catboost(train_auto, val_auto, common_auto, "auto_features")

    print("\nОцениваю на валидации...")
    val_metrics = evaluate_model(
        model_auto, val_auto, common_auto, cols, "auto_val"
    )

    print("\nСчитаю автоматические признаки для теста...")
    full_before_test = pd.concat([train, val], ignore_index=True)
    test_auto, test_auto_cols, _ = build_auto_features(
        test, full_before_test, cols, specs
    )

    test_common_auto = list(set(common_auto) & set(test_auto_cols))
    print(f"\nОцениваю на тесте...")
    test_metrics = evaluate_model(
        model_auto, test_auto, test_common_auto, cols, "auto_test"
    )

    return model_auto, common_auto


def main():
    print("Загрузка данных...")
    train, val, test, cols = load_data_and_cols()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    baseline_model, baseline_cols = run_baseline(train, val, test, cols)

    auto_model, auto_cols = run_auto_features(train, val, test, cols)

    from train_and_evaluate import compare_results
    print("\n")
    compare_results()

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == "__main__":
    main()
