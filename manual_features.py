import pandas as pd
import numpy as np
import json
import os
from config import PROCESSED_DIR, MANUAL_TIME_WINDOWS


def load_columns():
    with open(os.path.join(PROCESSED_DIR, "columns.json"), "r") as f:
        return json.load(f)


def compute_user_stats(history, user_col, ts_col, current_time, window_days):
    cutoff = current_time - pd.Timedelta(days=window_days)
    recent = history[(history[ts_col] >= cutoff) & (history[ts_col] < current_time)]

    stats = recent.groupby(user_col).agg(
        user_event_count=(user_col, "size"),
    ).reset_index()
    stats.columns = [user_col, f"user_count_{window_days}d"]

    if "label" in recent.columns:
        mean_label = recent.groupby(user_col)["label"].mean().reset_index()
        mean_label.columns = [user_col, f"user_mean_label_{window_days}d"]
        stats = stats.merge(mean_label, on=user_col, how="left")

    if "watched_pct" in recent.columns:
        wp = recent.groupby(user_col)["watched_pct"].agg(["mean", "std"]).reset_index()
        wp.columns = [user_col,
                      f"user_mean_wp_{window_days}d",
                      f"user_std_wp_{window_days}d"]
        stats = stats.merge(wp, on=user_col, how="left")

    return stats


def compute_item_stats(history, item_col, ts_col, current_time, window_days):
    cutoff = current_time - pd.Timedelta(days=window_days)
    recent = history[(history[ts_col] >= cutoff) & (history[ts_col] < current_time)]

    stats = recent.groupby(item_col).agg(
        item_event_count=(item_col, "size"),
    ).reset_index()
    stats.columns = [item_col, f"item_count_{window_days}d"]

    if "label" in recent.columns:
        mean_label = recent.groupby(item_col)["label"].mean().reset_index()
        mean_label.columns = [item_col, f"item_mean_label_{window_days}d"]
        stats = stats.merge(mean_label, on=item_col, how="left")

    if "watched_pct" in recent.columns:
        wp = recent.groupby(item_col)["watched_pct"].agg(["mean", "std"]).reset_index()
        wp.columns = [item_col,
                      f"item_mean_wp_{window_days}d",
                      f"item_std_wp_{window_days}d"]
        stats = stats.merge(wp, on=item_col, how="left")

    return stats


def compute_user_item_stats(history, user_col, item_col, ts_col,
                            current_time, window_days):
    cutoff = current_time - pd.Timedelta(days=window_days)
    recent = history[(history[ts_col] >= cutoff) & (history[ts_col] < current_time)]

    pair_stats = recent.groupby([user_col, item_col]).size().reset_index(name=f"pair_count_{window_days}d")
    return pair_stats


def compute_user_genre_stats(history, user_col, ts_col,
                             current_time, window_days):
    if "main_genre" not in history.columns:
        return None

    cutoff = current_time - pd.Timedelta(days=window_days)
    recent = history[(history[ts_col] >= cutoff) & (history[ts_col] < current_time)]

    genre_stats = recent.groupby([user_col, "main_genre"]).size().reset_index(
        name=f"user_genre_count_{window_days}d"
    )
    return genre_stats


def build_manual_features(target_df, history_df, cols):
    user_col = cols["user_id"]
    item_col = cols["item_id"]
    ts_col = cols["timestamp"]

    wp_col = cols.get("watched_pct", None)

    result = target_df[[user_col, item_col, ts_col, "label"]].copy()

    if wp_col and wp_col != "watched_pct" and wp_col in history_df.columns:
        history_df = history_df.rename(columns={wp_col: "watched_pct"})
    if wp_col and wp_col != "watched_pct" and wp_col in result.columns:
        result = result.rename(columns={wp_col: "watched_pct"})

    current_time = target_df[ts_col].max()

    feature_dfs_user = []
    feature_dfs_item = []
    feature_dfs_pair = []
    feature_dfs_genre = []

    for w in MANUAL_TIME_WINDOWS:
        user_stats = compute_user_stats(history_df, user_col, ts_col, current_time, w)
        feature_dfs_user.append(user_stats)

        item_stats = compute_item_stats(history_df, item_col, ts_col, current_time, w)
        feature_dfs_item.append(item_stats)

        pair_stats = compute_user_item_stats(history_df, user_col, item_col, ts_col, current_time, w)
        feature_dfs_pair.append(pair_stats)

        genre_stats = compute_user_genre_stats(history_df, user_col, ts_col, current_time, w)
        if genre_stats is not None:
            feature_dfs_genre.append(genre_stats)

    for df in feature_dfs_user:
        result = result.merge(df, on=user_col, how="left")

    for df in feature_dfs_item:
        result = result.merge(df, on=item_col, how="left")

    for df in feature_dfs_pair:
        result = result.merge(df, on=[user_col, item_col], how="left")

    if feature_dfs_genre and "main_genre" in result.columns:
        for df in feature_dfs_genre:
            result = result.merge(df, on=[user_col, "main_genre"], how="left")
    elif feature_dfs_genre and "main_genre" in target_df.columns:
        result["main_genre"] = target_df["main_genre"].values
        for df in feature_dfs_genre:
            result = result.merge(df, on=[user_col, "main_genre"], how="left")

    cat_features_to_add = []
    for c in ["age", "sex", "income", "kids_flg"]:
        if c in target_df.columns:
            result[c] = target_df[c].values
            cat_features_to_add.append(c)

    for c in ["release_year", "content_type", "for_kids", "age_rating"]:
        if c in target_df.columns:
            result[c] = target_df[c].values
            cat_features_to_add.append(c)

    if "total_dur" in target_df.columns:
        result["total_dur"] = target_df["total_dur"].values

    result = result.fillna(0)

    feature_cols = [c for c in result.columns
                    if c not in [user_col, item_col, ts_col, "label"]]

    return result, feature_cols


def build_features_by_periods(target_df, full_history, cols):
    user_col = cols["user_id"]
    item_col = cols["item_id"]
    ts_col = cols["timestamp"]

    target_df = target_df.copy()
    target_df[ts_col] = pd.to_datetime(target_df[ts_col])
    full_history = full_history.copy()
    full_history[ts_col] = pd.to_datetime(full_history[ts_col])

    history_before = full_history[full_history[ts_col] < target_df[ts_col].min()]

    result, feature_cols = build_manual_features(target_df, history_before, cols)
    return result, feature_cols


if __name__ == "__main__":
    cols = load_columns()
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    print(f"Train shape: {train.shape}")
    print("Columns:", cols)
