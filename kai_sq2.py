import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def load_results(train_file="rsf_train_results.csv", test_file="rsf_survival_prediction_results.csv"):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    return train_df, test_df
def compute_chi_square(train_df, test_df):
    train_positive = np.sum(train_df["True_Composite_Event"] == 1)
    train_negative = np.sum(train_df["True_Composite_Event"] == 0)

    test_positive = np.sum(test_df["True_Composite_Event"] == 1)
    test_negative = np.sum(test_df["True_Composite_Event"] == 0)
    contingency_table = np.array([[train_positive, train_negative],
                                  [test_positive, test_negative]])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"✅ 训练集 vs. 测试集 卡方检验 统计量: {chi2_stat:.4f}, p 值: {p_value:.4f}")
    print(f"✅ 训练集类别分布: {{1: {train_positive}, 0: {train_negative}}}")
    print(f"✅ 测试集类别分布: {{1: {test_positive}, 0: {test_negative}}}")
    with open("chi_square_results.txt", "w") as f:
        f.write(f"Chi-square Statistic: {chi2_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Train Class Distribution: {{1: {train_positive}, 0: {train_negative}}}\n")
        f.write(f"Test Class Distribution: {{1: {test_positive}, 0: {test_negative}}}\n")

    print("✅ 结果已保存至 chi_square_results.txt")

if __name__ == "__main__":
    train_df, test_df = load_results()
    compute_chi_square(train_df, test_df)
