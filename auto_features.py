import pandas as pd
import numpy as np
import json
import os
from config import PROCESSED_DIR, AUTO_TIME_WINDOWS, AUTO_AGGREGATIONS, TOP_N_AUTO_FEATURES
from tqdm import tqdm


def load_columns():
    with open(os.path.join(PROCESSED_DIR, "columns.json"), "r") as f:
        return json.load(f)


def generate_feature_specs(cols, df_columns):
    specs = []

    group_keys = []
    group_keys.append(("user", cols["user_id"]))
    group_keys.append(("item", cols["item_id"]))

    if "main_genre" in df_columns:
        group_keys.append(("user_genre", [cols["user_id"], "main_genre"]))
    if "content_type" in df_columns:
        group_keys.append(("user_ctype", [cols["user_id"], "content_type"]))

    pair_key = ("pair", [cols["user_id"], cols["item_id"]])
    group_keys.append(pair_key)

    agg_targets = ["__count__"]
    if "watched_pct" in df_columns:
        agg_targets.append("watched_pct")
    if "total_dur" in df_columns:
        agg_targets.append("total_dur")

    for group_name, group_col in group_keys:
        for window in AUTO_TIME_WINDOWS:
            for agg_target in agg_targets:
                if agg_target == "__count__":
                    spec = {
                        "name": f"auto_{group_name}_count_{window}d",
                        "group_col": group_col,
                        "window_days": window,
                        "agg_target": None,
                        "agg_func": "count",
                    }
                    specs.append(spec)
                else:
                    for agg_func in AUTO_AGGREGATIONS:
                        if agg_func == "count":
                            continue
                        spec = {
                            "name": f"auto_{group_name}_{agg_func}_{agg_target}_{window}d",
                            "group_col": group_col,
                            "window_days": window,
                            "agg_target": agg_target,
                            "agg_func": agg_func,
                        }
                        specs.append(spec)

    return specs


def compute_single_feature(history, spec, ts_col, current_time):
    window_days = spec["window_days"]
    cutoff = current_time - pd.Timedelta(days=window_days)
    recent = history[(history[ts_col] >= cutoff) & (history[ts_col] < current_time)]

    if len(recent) == 0:
        return None

    group_col = spec["group_col"]
    agg_func = spec["agg_func"]
    agg_target = spec["agg_target"]
    feat_name = spec["name"]

    if isinstance(group_col, str):
        group_col_list = [group_col]
    else:
        group_col_list = list(group_col)

    if agg_func == "count":
        result = recent.groupby(group_col_list).size().reset_index(name=feat_name)
    else:
        if agg_target not in recent.columns:
            return None
        result = recent.groupby(group_col_list)[agg_target].agg(agg_func).reset_index()
        result.columns = group_col_list + [feat_name]

    return result


def build_auto_features(target_df, history_df, cols, specs=None):
    user_col = cols["user_id"]
    item_col = cols["item_id"]
    ts_col = cols["timestamp"]

    target_df = target_df.copy()
    target_df[ts_col] = pd.to_datetime(target_df[ts_col])
    history_df = history_df.copy()
    history_df[ts_col] = pd.to_datetime(history_df[ts_col])

    history_before = history_df[history_df[ts_col] < target_df[ts_col].min()]
    current_time = target_df[ts_col].max()

    result = target_df[[user_col, item_col, ts_col, "label"]].copy()

    if "main_genre" in target_df.columns:
        result["main_genre"] = target_df["main_genre"].values
    if "content_type" in target_df.columns:
        result["content_type"] = target_df["content_type"].values

    if specs is None:
        specs = generate_feature_specs(cols, history_before.columns)

    print(f"Генерация {len(specs)} автоматических признаков...")

    computed_features = []
    for spec in tqdm(specs):
        feat_df = compute_single_feature(history_before, spec, ts_col, current_time)
        if feat_df is not None:
            group_col = spec["group_col"]
            if isinstance(group_col, str):
                merge_keys = [group_col]
            else:
                merge_keys = list(group_col)

            available_keys = [k for k in merge_keys if k in result.columns]
            if len(available_keys) == len(merge_keys):
                result = result.merge(feat_df, on=merge_keys, how="left")
                computed_features.append(spec["name"])

    cat_features_to_add = []
    for c in ["age", "sex", "income", "kids_flg",
              "release_year", "content_type", "for_kids", "age_rating"]:
        if c in target_df.columns and c not in result.columns:
            result[c] = target_df[c].values
            cat_features_to_add.append(c)

    result = result.fillna(0)

    feature_cols = [c for c in result.columns
                    if c not in [user_col, item_col, ts_col, "label"]]

    return result, feature_cols, specs


def select_features_by_importance(X_train, y_train, feature_cols, top_n):
    from lightgbm import LGBMClassifier

    print(f"Отбор топ-{top_n} признаков по важности...")

    numeric_cols = []
    for c in feature_cols:
        if X_train[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            numeric_cols.append(c)

    if len(numeric_cols) == 0:
        return feature_cols[:top_n]

    X_num = X_train[numeric_cols].values
    y = y_train.values

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_num, y)

    importances = model.feature_importances_
    importance_pairs = list(zip(numeric_cols, importances))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    selected = [p[0] for p in importance_pairs[:top_n]]

    print("Топ-10 признаков по важности:")
    for name, imp in importance_pairs[:10]:
        print(f"  {name}: {imp}")

    return selected


if __name__ == "__main__":
    cols = load_columns()
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    specs = generate_feature_specs(cols, train.columns)
    print(f"Сгенерировано {len(specs)} спецификаций признаков")
    for s in specs[:10]:
        print(f"  {s['name']}")
