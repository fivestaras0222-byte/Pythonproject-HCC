import os
import joblib
import numpy as np
import pandas as pd
import re  # Import regex for numeric extraction
import shap
import matplotlib.pyplot as plt
import re  # Import regex for numeric extraction
import matplotlib as mpl
import matplotlib.cm as cm
plt.switch_backend("macosx")
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
from lifelines import CoxPHFitter
def extract_numeric(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        match = re.search(r"[\d\.]+", str(x))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return np.nan
        else:
            return np.nan


def compute_p_values(model_path="newbest_rsf_model.pkl",
                     feature_csv="selected_features1_testuse.csv",
                     data_file="datahx1.csv",
                     output_csv="top_20_features_p_values.csv",
                     output_dir="plots_TEST"):
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到 RSF 模型 {model_path}，请先运行 rsf_model.py 进行训练。")
        return

    rsf = joblib.load(model_path)
    print(f"✅ 已加载 RSF 模型: {model_path}")
    df = pd.read_csv(data_file)
    selected_features_df = pd.read_csv(feature_csv)
    feature_mapping = dict(
        zip(selected_features_df["Selected_Feature_translated"], selected_features_df["Selected_Feature"]))
    print(feature_mapping)

    reverse_feature_mapping = {v: k for k, v in feature_mapping.items()}
    print("Model feature names:", rsf.feature_names_in_)

    print("Dataframe columns:", df.columns.tolist())
    df = df.rename(columns=feature_mapping)
    print("Mapped feature names:", df.columns.tolist())
    X = df[rsf.feature_names_in_]
    X = X.map(extract_numeric)

    X_sample = X.sample(n=min(60, len(X)), random_state=42).copy()
    print(X_sample.columns)

    try:
        explainer = shap.Explainer(rsf.predict, X_sample)
        shap_values = explainer(X_sample)
    except Exception as e:
        print(f"FAIL: {e}")
        return
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': X_sample.columns,
        'SHAP Importance': shap_importance
    })
    # X_sample.columns = [reverse_feature_mapping.get(col, col) for col in X_sample.columns]
    print(shap_importance_df)
    thr = 15
    top_5_features = shap_importance_df.sort_values(by='SHAP Importance', ascending=False).head(thr)
    top_5_feature_names = top_5_features['Feature'].values
    selected_shap_values = shap_values[:, top_5_feature_names]
    plt.figure(figsize=(12, 6))
    shap.summary_plot(selected_shap_values, X_sample[top_5_feature_names], plot_type="bar", show=False, max_display=thr)

    labels = [reverse_feature_mapping.get(col, col) for col in top_5_feature_names]
    plt.xlabel("Feature Impact on Model Output", fontsize=14)
    plt.ylabel("Feature Name", fontsize=14)
    plt.yticks(ticks=range(len(labels)), labels=labels[::-1], rotation=0, fontsize=10)
    plt.xticks(fontsize=12)
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=300)
    plt.close()
    plt.figure(figsize=(12, 6))
    shap.summary_plot(selected_shap_values, X_sample[top_5_feature_names], show=False)
    plt.xlabel("Feature Impact on Model Output", fontsize=14)
    plt.ylabel("Feature Name", fontsize=14)
    plt.yticks(ticks=range(len(labels)), labels=labels[::-1], rotation=0, fontsize=10)
    plt.xticks(fontsize=12)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300)
    plt.close()
    print("✅ 开始生成 decision plot (验证集所有样本)...")
    TIME_POINT = 730
    def predict_fn(X_in):
        if isinstance(X_in, np.ndarray):
            X_df = pd.DataFrame(X_in, columns=X.columns)
        else:
            X_df = X_in.copy()
        X_df = X_df.map(extract_numeric)

        surv_funcs = rsf.predict_survival_function(X_df)
        preds = np.array([1.0 - fn(TIME_POINT) for fn in surv_funcs], dtype=float)
        return preds
    background = X.sample(n=min(60, len(X)), random_state=0)
    explainer_dec = shap.PermutationExplainer(predict_fn, background)
    shap_values_all = explainer_dec(X)

    y_pred_all = predict_fn(X)
    base_value = float(np.mean(y_pred_all))
    top_features = list(top_5_feature_names)
    shap_values_subset = shap_values_all[:, top_features]
    feature_names_subset = [reverse_feature_mapping.get(c, c) for c in top_features]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    shap.decision_plot(
        base_value,
        shap_values_subset.values,
        feature_names=feature_names_subset,
        show=False,
        alpha=0.25
    )
    fig = plt.gcf()
    if len(fig.axes) > 1:
        fig.axes[1].set_xticklabels([])
        fig.axes[1].set_xticks([])
    plt.title("SHAP Decision Plot (Validation Set)", fontsize=14)
    plt.xlabel("Model Output", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    ax = plt.gca()
    ax.set_ylim(-0.5, len(top_features) - 0.5)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "shap_decision_plot_all.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
if __name__ == "__main__":
    compute_p_values()
