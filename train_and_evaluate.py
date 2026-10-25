import pandas as pd
import numpy as np
import os
import json
import pickle
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, CATBOOST_PARAMS, RANKING_K


def get_numeric_features(df, feature_cols):
    numeric_cols = []
    cat_cols = []
    for c in feature_cols:
        if df[c].dtype == object or df[c].dtype.name == "category":
            cat_cols.append(c)
        else:
            numeric_cols.append(c)
    return numeric_cols, cat_cols


def train_catboost(train_df, val_df, feature_cols, model_name="baseline"):
    numeric_cols, cat_cols = get_numeric_features(train_df, feature_cols)

    for c in cat_cols:
        train_df[c] = train_df[c].astype(str)
        val_df[c] = val_df[c].astype(str)

    train_pool = Pool(
        data=train_df[feature_cols],
        label=train_df["label"],
        cat_features=cat_cols if cat_cols else None,
    )
    val_pool = Pool(
        data=val_df[feature_cols],
        label=val_df["label"],
        cat_features=cat_cols if cat_cols else None,
    )

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    model_path = os.path.join(MODELS_DIR, f"{model_name}.cbm")
    model.save_model(model_path)
    print(f"Модель сохранена: {model_path}")

    return model


def predict_proba(model, df, feature_cols):
    numeric_cols, cat_cols = get_numeric_features(df, feature_cols)
    for c in cat_cols:
        df[c] = df[c].astype(str)
    pool = Pool(data=df[feature_cols], cat_features=cat_cols if cat_cols else None)
    preds = model.predict_proba(pool)[:, 1]
    return preds


def compute_classification_metrics(y_true, y_pred):
    metrics = {}
    metrics["logloss"] = log_loss(y_true, y_pred)
    metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    return metrics


def ndcg_at_k(y_true, y_pred, k):
    order = np.argsort(-y_pred)
    y_sorted = np.array(y_true)[order]

    dcg = 0.0
    for i in range(min(k, len(y_sorted))):
        dcg += y_sorted[i] / np.log2(i + 2)

    ideal_order = np.argsort(-np.array(y_true))
    y_ideal = np.array(y_true)[ideal_order]

    idcg = 0.0
    for i in range(min(k, len(y_ideal))):
        idcg += y_ideal[i] / np.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(y_true, y_pred, k):
    order = np.argsort(-y_pred)
    top_k_labels = np.array(y_true)[order[:k]]

    total_relevant = np.sum(y_true)
    if total_relevant == 0:
        return 0.0

    return np.sum(top_k_labels) / total_relevant


def compute_ranking_metrics(df, user_col, y_true_col, y_pred_col, k_values):
    metrics = {}
    for k in k_values:
        ndcg_scores = []
        recall_scores = []

        users = df[user_col].unique()
        for user in users:
            user_mask = df[user_col] == user
            user_df = df[user_mask]

            if len(user_df) < 2:
                continue

            y_true = user_df[y_true_col].values
            y_pred = user_df[y_pred_col].values

            if y_true.sum() == 0:
                continue

            ndcg_scores.append(ndcg_at_k(y_true, y_pred, k))
            recall_scores.append(recall_at_k(y_true, y_pred, k))

        if ndcg_scores:
            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
        else:
            metrics[f"ndcg@{k}"] = 0.0

        if recall_scores:
            metrics[f"recall@{k}"] = np.mean(recall_scores)
        else:
            metrics[f"recall@{k}"] = 0.0

    return metrics


def evaluate_model(model, test_df, feature_cols, cols, model_name="baseline"):
    user_col = cols["user_id"]

    preds = predict_proba(model, test_df.copy(), feature_cols)

    y_true = test_df["label"].values
    clf_metrics = compute_classification_metrics(y_true, preds)

    eval_df = test_df[[user_col, "label"]].copy()
    eval_df["pred"] = preds

    ranking_metrics = compute_ranking_metrics(
        eval_df, user_col, "label", "pred", RANKING_K
    )

    all_metrics = {}
    all_metrics.update(clf_metrics)
    all_metrics.update(ranking_metrics)

    print(f"\n{'='*50}")
    print(f"Результаты модели: {model_name}")
    print(f"{'='*50}")
    for name, value in all_metrics.items():
        print(f"  {name}: {value:.4f}")

    results_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


def compare_results():
    results_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_metrics.json")]

    all_results = {}
    for fname in results_files:
        model_name = fname.replace("_metrics.json", "")
        with open(os.path.join(RESULTS_DIR, fname), "r") as f:
            all_results[model_name] = json.load(f)

    if not all_results:
        print("Нет результатов для сравнения")
        return

    print(f"\n{'='*70}")
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print(f"{'='*70}")

    metrics_names = list(list(all_results.values())[0].keys())
    model_names = list(all_results.keys())

    header = f"{'Метрика':<20}"
    for m in model_names:
        header += f"{m:<20}"
    print(header)
    print("-" * 70)

    for metric in metrics_names:
        row = f"{metric:<20}"
        for m in model_names:
            val = all_results[m].get(metric, 0)
            row += f"{val:<20.4f}"
        print(row)

    comparison = pd.DataFrame(all_results)
    comparison.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"))
    print(f"\nТаблица сравнения сохранена: {os.path.join(RESULTS_DIR, 'comparison.csv')}")

    return all_results
