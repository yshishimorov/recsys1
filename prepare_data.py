import pandas as pd
import numpy as np
import os
from config import RAW_DIR, PROCESSED_DIR, N_SAMPLE_USERS, RANDOM_SEED
from config import TRAIN_RATIO, VAL_RATIO, WATCHED_THRESHOLD


def load_raw_data():
    interactions = pd.read_csv(os.path.join(RAW_DIR, "interactions.csv"))
    users = pd.read_csv(os.path.join(RAW_DIR, "users.csv"))
    items = pd.read_csv(os.path.join(RAW_DIR, "items.csv"))

    print(f"Interactions shape: {interactions.shape}")
    print(f"Users shape: {users.shape}")
    print(f"Items shape: {items.shape}")
    print()
    print("Interactions columns:", list(interactions.columns))
    print("Users columns:", list(users.columns))
    print("Items columns:", list(items.columns))

    return interactions, users, items


def detect_columns(interactions):
    cols = {}

    for c in interactions.columns:
        cl = c.lower()
        if "user" in cl:
            cols["user_id"] = c
        elif "item" in cl:
            cols["item_id"] = c
        elif "watch" in cl and ("dt" in cl or "date" in cl or "time" in cl):
            cols["timestamp"] = c
        elif "watch" in cl and "pct" in cl:
            cols["watched_pct"] = c
        elif "dur" in cl and "total" in cl:
            cols["total_dur"] = c

    if "timestamp" not in cols:
        for c in interactions.columns:
            cl = c.lower()
            if "time" in cl or "date" in cl or "dt" in cl:
                cols["timestamp"] = c
                break

    print("Определённые столбцы:", cols)
    return cols


def sample_users(interactions, cols, n_users):
    all_users = interactions[cols["user_id"]].unique()
    print(f"Всего уникальных пользователей: {len(all_users)}")

    if len(all_users) <= n_users:
        print(f"Берём всех пользователей")
        return interactions

    np.random.seed(RANDOM_SEED)
    sampled_users = np.random.choice(all_users, size=n_users, replace=False)
    sampled = interactions[interactions[cols["user_id"]].isin(sampled_users)].copy()
    print(f"После сэмплирования {n_users} пользователей: {sampled.shape[0]} строк")
    return sampled


def prepare_timestamps(interactions, cols):
    ts_col = cols["timestamp"]
    interactions[ts_col] = pd.to_datetime(interactions[ts_col])
    interactions = interactions.sort_values(ts_col).reset_index(drop=True)
    return interactions


def create_target(interactions, cols):
    if "watched_pct" in cols:
        wp_col = cols["watched_pct"]
        vals = interactions[wp_col]
        if vals.max() > 1:
            interactions["label"] = (vals >= WATCHED_THRESHOLD * 100).astype(int)
        else:
            interactions["label"] = (vals >= WATCHED_THRESHOLD).astype(int)
    else:
        interactions["label"] = 1

    print(f"Распределение label:")
    print(interactions["label"].value_counts())
    return interactions


def temporal_split(interactions, cols):
    ts_col = cols["timestamp"]
    interactions = interactions.sort_values(ts_col).reset_index(drop=True)

    n = len(interactions)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = interactions.iloc[:train_end].copy()
    val = interactions.iloc[train_end:val_end].copy()
    test = interactions.iloc[val_end:].copy()

    print(f"Train: {len(train)} строк, "
          f"от {train[ts_col].min()} до {train[ts_col].max()}")
    print(f"Val:   {len(val)} строк, "
          f"от {val[ts_col].min()} до {val[ts_col].max()}")
    print(f"Test:  {len(test)} строк, "
          f"от {test[ts_col].min()} до {test[ts_col].max()}")

    return train, val, test


def merge_metadata(interactions, users, items, cols):
    user_col = cols["user_id"]
    item_col = cols["item_id"]

    users_col_map = {}
    for c in users.columns:
        cl = c.lower()
        if "user" in cl and ("id" in cl):
            users_col_map["user_id"] = c

    items_col_map = {}
    for c in items.columns:
        cl = c.lower()
        if "item" in cl and ("id" in cl):
            items_col_map["item_id"] = c

    if users_col_map:
        interactions = interactions.merge(
            users,
            left_on=user_col,
            right_on=users_col_map["user_id"],
            how="left",
            suffixes=("", "_user_meta")
        )

    if items_col_map:
        interactions = interactions.merge(
            items,
            left_on=item_col,
            right_on=items_col_map["item_id"],
            how="left",
            suffixes=("", "_item_meta")
        )

    return interactions


def standardize_watched_pct(interactions, cols):
    if "watched_pct" in cols:
        wp_col = cols["watched_pct"]
        if wp_col != "watched_pct":
            interactions["watched_pct"] = interactions[wp_col]
        if interactions["watched_pct"].max() > 1:
            interactions["watched_pct"] = interactions["watched_pct"] / 100.0
    return interactions


def extract_genre_features(df):
    if "genres" in df.columns:
        first_genre = df["genres"].astype(str).str.split(",").str[0].str.strip()
        df["main_genre"] = first_genre
    return df


def main():
    interactions, users, items = load_raw_data()
    cols = detect_columns(interactions)

    interactions = sample_users(interactions, cols, N_SAMPLE_USERS)
    interactions = prepare_timestamps(interactions, cols)
    interactions = create_target(interactions, cols)

    interactions = merge_metadata(interactions, users, items, cols)
    interactions = standardize_watched_pct(interactions, cols)
    interactions = extract_genre_features(interactions)

    train, val, test = temporal_split(interactions, cols)

    train.to_parquet(os.path.join(PROCESSED_DIR, "train.parquet"), index=False)
    val.to_parquet(os.path.join(PROCESSED_DIR, "val.parquet"), index=False)
    test.to_parquet(os.path.join(PROCESSED_DIR, "test.parquet"), index=False)

    import json
    with open(os.path.join(PROCESSED_DIR, "columns.json"), "w") as f:
        json.dump(cols, f, indent=2, default=str)

    print()
    print("Данные подготовлены и сохранены в", PROCESSED_DIR)


if __name__ == "__main__":
    main()
