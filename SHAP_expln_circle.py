import os
import joblib
import numpy as np
import pandas as pd
import shap
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

plt.switch_backend("macosx")
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
#
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


def plot_radial_shap_with_scatter(shap_values, X_sample, reverse_feature_mapping=None,
                                   output_dir="shap_plots", top_n=15):
    print("🎯 正在绘制极坐标 SHAP 柱状图 + 散点...")

    shap_importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = X_sample.columns.tolist()
    # shap_df = pd.DataFrame({
    #     "feature": feature_names,
    #     "importance": shap_importance
    # }).sort_values(by="importance", ascending=False).head(top_n)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "importance": shap_importance
    }).sample(n=top_n, random_state=42)

    top_features = shap_df["feature"].tolist()
    radii = shap_df["importance"].values
    angles = np.linspace(0, 2 * np.pi, top_n, endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')

    inner_radius = 1
    bar_colors = plt.cm.Reds((radii - radii.min()) / (radii.max() - radii.min() + 1e-9))
    bar_width = 2 * np.pi / top_n * 0.5

    bars = ax.bar(angles, radii, width=bar_width, bottom=inner_radius,
                  color=bar_colors, edgecolor='none', linewidth=0.6)

    scatter_base = inner_radius + 17
    cmap = plt.get_cmap("RdBu_r")
    norm = mpl.colors.Normalize(vmin=X_sample[top_features].values.min(),
                                vmax=X_sample[top_features].values.max())

    # for angle, feature in zip(angles, top_features):
    #     shap_vals = shap_values[:, feature].values
    #     feat_vals = X_sample[feature].values
    #
    #     theta_pos = np.ones_like(shap_vals) * angle + np.random.normal(scale=0.02, size=len(shap_vals))
    #     r_pos = scatter_base + shap_vals
    #     c = cmap(norm(feat_vals))
    #
    #     ax.scatter(theta_pos, r_pos, c=c, s=20, alpha=0.7, edgecolors='none')
    for angle, feature in zip(angles, top_features):
            shap_vals = shap_values[:, feature].values * 0.3
            feat_vals = X_sample[feature].values

            r = scatter_base + shap_vals
            theta = np.ones_like(r) * angle + np.random.normal(scale=0.02, size=len(r))
            norm = mpl.colors.Normalize(vmin=np.nanmin(feat_vals), vmax=np.nanmax(feat_vals))
            colors = plt.cm.RdBu_r(norm(feat_vals))

            ax.scatter(theta, r, c=colors, s=20, alpha=0.8, edgecolors='none')

    labels = [reverse_feature_mapping.get(f, f) if reverse_feature_mapping else f for f in top_features]
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)

    max_height = scatter_base + np.max(np.abs(shap_values.values[:, [X_sample.columns.get_loc(f) for f in top_features]]))*0.2
    yticks = np.linspace(inner_radius, max_height, 5)
    yticklabels = [f"{(y - inner_radius):.0f}" for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_ylim(0, max_height)
    y_axis_angle = np.pi / 2
    ax.plot([y_axis_angle, y_axis_angle], [inner_radius, max_height],
            color='black', linewidth=0.7, linestyle='-')
    ax.plot(np.linspace(0, 2 * np.pi, 500), [scatter_base] * 500,
            color='gray', linewidth=0.8, linestyle='--')
    fill_theta = np.linspace(0, 2 * np.pi, 500)
    fill_r_min = 0
    fill_r_max = max_height
    ax.fill_between(fill_theta, fill_r_min, fill_r_max,
                    color=(0.95294117647,0.94117647059,0.85882352941), alpha=0.2)
    tick_length = radii.max() * 0.05
    for angle in angles:
        ax.plot([angle, angle], [max_height - tick_length, max_height], color='black', linewidth=1)

    ax.spines['polar'].set_visible(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(False)
    ax.tick_params(axis='x', pad=15)


    plt.title("Radial SHAP Bars + Scatter", fontsize=15, y=1.08)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outpath = os.path.join(output_dir, "radial_shap_with_scatter.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def compute_and_plot_shap(model_path="newbest_rsf_model.pkl",
                          feature_csv="selected_features1_testuse.csv",
                          data_file="datahx1.csv",
                          output_dir="shap_plots",
                          top_n=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rsf = joblib.load(model_path)
    print(f"✅ 模型已加载: {model_path}")

    df = pd.read_csv(data_file)
    selected_features_df = pd.read_csv(feature_csv)

    feature_mapping = dict(zip(selected_features_df["Selected_Feature_translated"],
                               selected_features_df["Selected_Feature"]))
    reverse_feature_mapping = {v: k for k, v in feature_mapping.items()}

    df = df.rename(columns=feature_mapping)
    X = df[rsf.feature_names_in_].map(extract_numeric)
    X_sample = X.sample(n=min(60, len(X)), random_state=42).copy()

    explainer = shap.Explainer(rsf.predict, X_sample)
    shap_values = explainer(X_sample)

    print("✅ SHAP 值计算完成，开始绘图...")
    plot_radial_shap_with_scatter(shap_values, X_sample, reverse_feature_mapping, output_dir, top_n)
if __name__ == "__main__":
    compute_and_plot_shap()