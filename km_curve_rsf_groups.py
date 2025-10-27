import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
plt.switch_backend('macosx')

def plot_km_curve(df, time_col, event_col, group_col, title):
    df = df[df["risk_group"].isin(["High Risk", "Low Risk"])]

    kmf = KaplanMeierFitter()

    for name, grouped_df in df.groupby(group_col):
        kmf.fit(grouped_df[time_col], grouped_df[event_col], label=name)
        kmf.plot_survival_function()

    plt.title(title)
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(title)


def risk_stratification_km(df, pred_prob_col, time_col, event_col, threshold=0.6, title=""):
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
    df = df.dropna(subset=[time_col, event_col, pred_prob_col])
    df = df[np.isfinite(df[time_col]) & np.isfinite(df[event_col]) & np.isfinite(df[pred_prob_col])]
    df["risk_group"] = np.where(df[pred_prob_col] >= threshold, "High Risk", "Low Risk")

    # Log-rank test
    results = logrank_test(
        df.loc[df["risk_group"] == "High Risk", time_col],
        df.loc[df["risk_group"] == "Low Risk", time_col],
        event_observed_A=df.loc[df["risk_group"] == "High Risk", event_col],
        event_observed_B=df.loc[df["risk_group"] == "Low Risk", event_col],
    )
    print(f"\n📊 {title}")
    print(f"Log-rank test p-value: {results.p_value:.4f}")

    # Plot KM curve
    plot_km_curve(df, time_col, event_col, "risk_group", title)


if __name__ == "__main__":
    # 文件路径
    train_file = "rsf_newtest_train_results.csv"
    test_file = "rsf_newtest_results.csv"

    # 读取数据
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # 添加时间和事件列（来自你原始数据）
    data_train = pd.read_csv("data6_副本.csv")
    data_test = pd.read_csv("datahx1.csv")

    df_train["时间段"] = data_train.loc[df_train.index, "时间段"].values
    df_train["术后复发（是=1，否=0）"] = data_train.loc[df_train.index, "术后复发（是=1，否=0）"].values

    df_test["时间段"] = data_test.loc[df_test.index, "时间段"].values
    df_test["术后复发（是=1，否=0）"] = data_test.loc[df_test.index, "术后复发（是=1，否=0）"].values

    # 分析
    # risk_stratification_km(df_train, pred_prob_col="Predicted_Event_Probability",
    #                        time_col="时间段", event_col="术后复发（是=1，否=0）",
    #                        threshold=0.6, title="Training Set Survival by Risk Group")

    risk_stratification_km(df_test, pred_prob_col="Predicted_Event_Probability",
                           time_col="时间段", event_col="术后复发（是=1，否=0）",
                           threshold=0.6, title="Validation Set Survival by Risk Group")
