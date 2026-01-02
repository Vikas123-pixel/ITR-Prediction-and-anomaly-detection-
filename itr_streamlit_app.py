"""
Streamlit app for ITR Prediction & Anomaly Detection

Features:
- Upload CSV or use provided sample
- Select target column
- Train regression model (RandomForest) with simple preprocessing
- Show regression metrics, feature importance
- Run anomaly detection (IsolationForest + LOF) and view/download top anomalies
- Save trained models and reports

Run:
    pip install streamlit pandas numpy scikit-learn joblib
    streamlit run itr_streamlit_app.py

Author: ChatGPT (GPT-5 Thinking mini)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import io
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics

st.set_page_config(page_title="ITR: Predict & Detect", layout="wide")

# ----------------- Utilities -----------------
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "gross_income": np.random.lognormal(mean=11, sigma=0.5, size=n).round(2),
        "deductions_80c": np.random.uniform(0, 150000, size=n).round(2),
        "deductions_other": np.random.uniform(0, 250000, size=n).round(2),
        "tds": np.random.uniform(0, 200000, size=n).round(2),
        "advance_tax": np.random.uniform(0, 200000, size=n).round(2),
        "age": np.random.randint(21, 70, size=n),
        "assessment_year": np.random.choice(["AY2022-23","AY2023-24","AY2024-25"], size=n),
        "state": np.random.choice(["MH","DL","KA","TN","GJ","WB","RJ","UP"], size=n),
        "occupation": np.random.choice(["Salaried","Business","Professional","Retired"], size=n),
        "filing_type": np.random.choice(["Original","Revised"], size=n),
        "prev_gross_income": np.random.lognormal(mean=10.9, sigma=0.55, size=n).round(2),
    })
    eff = (df["gross_income"] - (df["deductions_80c"] + df["deductions_other"]).clip(lower=0)).clip(lower=0)
    tax = (
        0.0 * (eff.clip(upper=250000))
        + 0.05 * ((eff - 250000).clip(lower=0, upper=250000))
        + 0.2 * ((eff - 500000).clip(lower=0, upper=500000))
        + 0.3 * ((eff - 1000000).clip(lower=0))
    )
    itr_amount = (tax - 0.8 * df["tds"] - 0.6 * df["advance_tax"] + np.random.normal(0, 10000, size=n)).round(2)
    df["itr_amount"] = itr_amount
    return df


def infer_feature_types(df, target):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, categorical_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe


def evaluate_regression(y_true, y_pred):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = metrics.r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def fit_anomaly_models(X):
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X)
    iso_scores = -iso.score_samples(X)
    iso_labels = iso.predict(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto", novelty=False, n_jobs=-1)
    lof_labels = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_

    return (iso_scores, iso_labels), (lof_scores, lof_labels)


def df_to_download_link(df, name="data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    href = f"data:file/csv;base64,{b64}"
    return href


# ----------------- UI -----------------
st.title("TaxSense — ITR Prediction & Anomaly Detection")
st.sidebar.header("Data & Settings")

use_sample = st.sidebar.checkbox("Use sample dataset", value=True)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"]) if not use_sample else None

if use_sample:
    df = load_sample_data()
else:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV or enable 'Use sample dataset' to proceed.")
        st.stop()
st.write("### Preview data")
st.dataframe(df.head())

cols = df.columns.tolist()
default_target = "itr_amount" if "itr_amount" in cols else cols[-1]
target = st.sidebar.selectbox("Select target column (numeric)", cols, index=cols.index(default_target))

numeric_cols, categorical_cols = infer_feature_types(df, target)
st.sidebar.write(f"Numeric columns: {numeric_cols}")
st.sidebar.write(f"Categorical columns: {categorical_cols}")

train_btn = st.sidebar.button("Train Model")

if train_btn:
    with st.spinner("Training model — this may take a moment..."):
        y = df[target].astype(float)
        X = df.drop(columns=[target])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipe = build_pipeline(numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        metrics_dict = evaluate_regression(y_test, preds)

        st.success("Training complete")
        st.subheader("Regression metrics")
        st.json(metrics_dict)

        # Feature importance (if available)
        try:
            pre = pipe.named_steps["pre"]
            model = pipe.named_steps["model"]
            out_names = []
            for name, trans, cols_t in pre.transformers_:
                if name == "num":
                    out_names.extend(cols_t)
                elif name == "cat":
                    ohe = trans
                    try:
                        cat_names = ohe.get_feature_names_out(cols_t).tolist()
                    except Exception:
                        try:
                            cat_names = ohe.get_feature_names(cols_t).tolist()
                        except Exception:
                            cats = getattr(ohe, "categories_", [])
                            cat_names = []
                            for coln, cat_list in zip(cols_t, cats):
                                cat_names.extend([f"{coln}_{v}" for v in cat_list])
                    out_names.extend(cat_names)
            fi = getattr(model, "feature_importances_", None)
            if fi is not None:
                fi_df = pd.DataFrame({"feature": out_names, "importance": fi}).sort_values("importance", ascending=False)
                st.subheader("Feature importance")
                st.dataframe(fi_df.head(20))
                st.download_button("Download feature importance", fi_df.to_csv(index=False).encode("utf-8"), file_name="feature_importance.csv")
        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")

        # Anomaly detection
        X_full = pd.concat([X_train, X_test], axis=0)
        X_enc = pipe.named_steps["pre"].transform(X_full)
        X_enc = pd.DataFrame(X_enc)

        (iso_scores, iso_labels), (lof_scores, lof_labels) = fit_anomaly_models(X_enc)
        anomaly_df = pd.DataFrame({
            "iso_score": iso_scores,
            "iso_label": iso_labels,
            "lof_score": lof_scores,
            "lof_label": lof_labels,
        }, index=X_full.index)
        anomaly_df["combined_score"] = anomaly_df[["iso_score", "lof_score"]].mean(axis=1)
        anomaly_df = anomaly_df.sort_values("combined_score", ascending=False)

        joined = df.drop(columns=[target]).join(anomaly_df, how="left")
        st.subheader("Top anomalies")
        st.dataframe(joined.head(50))
        st.download_button("Download top anomalies", joined.head(500).to_csv(index=False).encode("utf-8"), file_name="anomaly_report_topk.csv")

        # Save model and iso
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        joblib.dump(pipe, out_dir / "itr_regressor.joblib")
        try:
            iso_model = IsolationForest(n_estimators=200, random_state=42)
            iso_model.fit(X_enc)
            joblib.dump(iso_model, out_dir / "isolation_forest.joblib")
        except Exception:
            pass

        st.info(f"Artifacts saved to {str(out_dir.resolve())}")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: TaxSense — ChatGPT")

st.sidebar.write("Need help? Upload your CSV and press 'Train Model'.")

# Footer: allow user to inspect full dataset
with st.expander("Show full dataset"):
    st.dataframe(df)
