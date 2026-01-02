"""
ITR Prediction & Anomaly Detection Pipeline (Python 3.9+)

Author: ChatGPT (GPT-5)
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
import joblib


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def infer_feature_types(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset columns")

    drop_cols = [c for c in df.columns if c != target and (df[c].isna().all() or df[c].nunique(dropna=True) <= 1)]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]
    return df, numeric_cols, categorical_cols


def build_regression_pipeline(numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        # Handle sklearn version differences
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, categorical_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe


def evaluate_regression(y_true, y_pred):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    # Manual RMSE calculation for compatibility with older sklearn
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = metrics.r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def fit_anomaly_models(X: pd.DataFrame, random_state=42):
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X)
    iso_scores = -iso.score_samples(X)
    iso_labels = iso.predict(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto", novelty=False, n_jobs=-1)
    lof_labels = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_

    return (iso, iso_scores, iso_labels), (lof, lof_scores, lof_labels)


def feature_importance(pipe: Pipeline, feature_names: list) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    out_names = []

    for name, trans, cols in pre.transformers_:
        if name == "num":
            out_names.extend(cols)
        elif name == "cat":
            ohe = trans
            try:
                cat_names = ohe.get_feature_names_out(cols).tolist()
            except Exception:
                try:
                    cat_names = ohe.get_feature_names(cols).tolist()
                except Exception:
                    cats = getattr(ohe, "categories_", [])
                    cat_names = []
                    for col, cat_list in zip(cols, cats):
                        cat_names.extend([f"{col}_{str(v)}" for v in cat_list])
            out_names.extend(cat_names)

    fi = getattr(model, "feature_importances_", None)
    if fi is None:
        return pd.DataFrame({"feature": out_names, "importance": np.nan})

    imp = pd.DataFrame({"feature": out_names, "importance": fi})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp


def main():
    parser = argparse.ArgumentParser(description="ITR prediction and anomaly detection pipeline")
    parser.add_argument("--data", type=str, default="itr_data_sample.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="itr_amount", help="Target column name (numeric)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--top_k_anomalies", type=int, default=50, help="How many top anomalies to save")
    parser.add_argument("--outputs", type=str, default=".", help="Root output directory")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_root = Path(args.outputs)
    models_dir = out_root / "models"
    reports_dir = out_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {data_path.resolve()}")
    df = load_data(data_path)

    df, numeric_cols, categorical_cols = infer_feature_types(df, args.target)
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    y = df[args.target].astype(float)
    X = df.drop(columns=[args.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    reg = build_regression_pipeline(numeric_cols, categorical_cols)
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)
    metrics_dict = evaluate_regression(y_test, pred)
    print("Regression metrics:", metrics_dict)

    joblib.dump(reg, models_dir / "itr_regressor.joblib")
    pd.DataFrame([metrics_dict]).to_csv(reports_dir / "regression_metrics.csv", index=False)

    try:
        imp = feature_importance(reg, X_train.columns.tolist())
        imp.to_csv(reports_dir / "feature_importance.csv", index=False)
    except Exception as e:
        print("Feature importance skipped:", e)

    X_full = pd.concat([X_train, X_test], axis=0)
    X_enc = reg.named_steps["pre"].transform(X_full)
    X_enc = pd.DataFrame(X_enc)

    (iso, iso_scores, iso_labels), (lof, lof_scores, lof_labels) = fit_anomaly_models(X_enc, random_state=args.random_state)

    anomaly_df = pd.DataFrame({
        "iso_score": iso_scores,
        "iso_label": iso_labels,
        "lof_score": lof_scores,
        "lof_label": lof_labels,
    }, index=X_full.index)

    anomaly_df["combined_score"] = anomaly_df[["iso_score", "lof_score"]].mean(axis=1)
    anomaly_df = anomaly_df.sort_values("combined_score", ascending=False)

    joined = df.drop(columns=[args.target]).join(anomaly_df, how="left")
    top_k = joined.head(args.top_k_anomalies)
    top_k.to_csv(reports_dir / "anomaly_report_topk.csv", index=False)

    joblib.dump(iso, models_dir / "isolation_forest.joblib")

    print("\n=== Done ===")
    print(f"Model saved: {models_dir / 'itr_regressor.joblib'}")
    print(f"Metrics CSV: {reports_dir / 'regression_metrics.csv'}")
    print(f"Feature importance CSV: {reports_dir / 'feature_importance.csv'}")
    print(f"Anomaly report CSV (top-k): {reports_dir / 'anomaly_report_topk.csv'}")


if __name__ == "__main__":
    main()
