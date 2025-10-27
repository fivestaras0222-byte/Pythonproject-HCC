import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import os
plt.switch_backend('macosx')
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
def plot_km_curve(df, time_col, event_col, group_col, title,
                  medians_dict, p_value, hr_ci_text, output_dir='km',
                  timepoints=None):
    df = df[df[group_col].isin(["High Risk", "Low Risk"])]
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8, 7))
    ax = plt.subplot(111)
    for name, grouped_df in df.groupby(group_col):
        kmf.fit(grouped_df[time_col], grouped_df[event_col], label=name)
        kmf.plot_survival_function(ci_show=True, ax=ax)
    info_text = f"Median RFS (days):\n"
    for grp, val in medians_dict.items():
        info_text += f"  {grp}: {val:.0f}\n"
    info_text += f"\nLog-rank P = {p_value:.4f}\n"
    info_text += f"{hr_ci_text}"

    plt.text(0.75, 0.63, info_text,
             transform=ax.transAxes,
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.title(title)
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    if timepoints is None:
        max_time = df[time_col].max()
        timepoints = np.linspace(0, 1500, 8, dtype=int)

    at_risk_counts = {}
    for name, grouped_df in df.groupby(group_col):
        kmf.fit(grouped_df[time_col], grouped_df[event_col], label=name)
        counts = [kmf.event_table.loc[kmf.event_table.index >= t, 'at_risk'].iloc[0]
                  if t <= kmf.event_table.index.max() else 0
                  for t in timepoints]
        at_risk_counts[name] = counts
    ypos = -0.15
    for i, (group, counts) in enumerate(at_risk_counts.items()):
        plt.text(-0.02, ypos - i*0.05, group, transform=ax.transAxes,
                 ha='right', fontsize=10)
        for j, t in enumerate(timepoints):
            plt.text(j/(len(timepoints)-1), ypos - i*0.05, str(counts[j]),
                     transform=ax.transAxes, ha='center', fontsize=10)
    for j, t in enumerate(timepoints):
        plt.text(j/(len(timepoints)-1), ypos + 0.05, str(int(t)),
                 transform=ax.transAxes, ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}.png'), dpi=300, bbox_inches="tight")
    plt.show()



def risk_stratification_km(df, pred_prob_col, time_col, event_col,
                           threshold=0.537, title=""):
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
    df = df.dropna(subset=[time_col, event_col, pred_prob_col])
    df = df[np.isfinite(df[time_col]) & np.isfinite(df[event_col]) & np.isfinite(df[pred_prob_col])]
    df["risk_group"] = np.where(df[pred_prob_col] >= threshold, "High Risk", "Low Risk")
    kmf = KaplanMeierFitter()
    medians = {}
    for name, grouped_df in df.groupby("risk_group"):
        kmf.fit(grouped_df[time_col], grouped_df[event_col])
        medians[name] = kmf.median_survival_time_
    results = logrank_test(
        df.loc[df["risk_group"] == "High Risk", time_col],
        df.loc[df["risk_group"] == "Low Risk", time_col],
        event_observed_A=df.loc[df["risk_group"] == "High Risk", event_col],
        event_observed_B=df.loc[df["risk_group"] == "Low Risk", event_col],
    )
    p_val = results.p_value
    cph_df = df[[time_col, event_col, "risk_group"]].copy()
    cph_df["risk_group"] = (cph_df["risk_group"] == "High Risk").astype(int)
    cph_df.columns = ["duration", "event", "high_risk"]

    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col="duration", event_col="event")

    summary = cph.summary
    hr = summary.loc["high_risk", "exp(coef)"]
    ci_lower = summary.loc["high_risk", "exp(coef) lower 95%"]
    ci_upper = summary.loc["high_risk", "exp(coef) upper 95%"]

    hr_ci_text = f"HR (High vs Low): {hr:.2f}\n95% CI: {ci_lower:.2f}–{ci_upper:.2f}"

    print(f"\n📊 {title}")
    print(f"Median RFS: {medians}")
    print(f"Log-rank P: {p_val:.4f}")
    print(hr_ci_text)
    plot_km_curve(df, time_col, event_col, "risk_group", title,
                  medians, p_val, hr_ci_text)


if __name__ == "__main__":
    train_file = "rsf_newtest_train_results.csv"
    test_file = "rsf_newtest_results_nostriate.csv"
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    data_train = pd.read_csv("data6_副本.csv")
    data_test = pd.read_csv("datahx1.csv")
    df_train["时间段"] = data_train.loc[df_train.index, "时间段"].values
    df_train["术后复发（是=1，否=0）"] = data_train.loc[df_train.index, "术后复发（是=1，否=0）"].values
    df_test["时间段"] = data_test.loc[df_test.index, "时间段"].values
    df_test["术后复发（是=1，否=0）"] = data_test.loc[df_test.index, "术后复发（是=1，否=0）"].values
    risk_stratification_km(df_train, pred_prob_col="Predicted_Event_Probability",
                           time_col="时间段", event_col="术后复发（是=1，否=0）",
                        title="Training Set Survival by Risk Group")
    risk_stratification_km(df_test, pred_prob_col="Predicted_Event_Probability",
                           time_col="时间段", event_col="术后复发（是=1，否=0）",
                         title="Validation Set Survival by Risk Group")
