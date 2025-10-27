import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

# ===== 1. 人工指定变量 =====
categorical_vars = ["性别_1", "是否合并肝炎_1", "是否合并肝硬化_1", "包膜是否受侵犯_未浸及",
                    "肿瘤是否巨块型分化_1", "是否开腹手术（1=开腹，无=腹腔镜手术）_1",
                    "肿瘤MVI_M0", "是否大范围切除_1", "child分期_A",
                    "AFP_less_400", "AFP_greater_400"
                    ]   # 分类变量
continuous_vars = ["年龄__y", "ALT", "AST", "TBIL", "R-GGT", "ALP", "ALB",
                   "PT", "INR", "WBC", "PLT", "肿瘤直径", "失血量", "ALBI"]  # 连续变量

# ===== 2. 读入数据 =====
train = pd.read_csv("data6_副本.csv")
val   = pd.read_csv("datahx1.csv")

# 只保留验证集里有的变量
common_vars = [col for col in val.columns if col in train.columns]

results = []

# ===== 3. 定义 SMD 计算函数 =====
def compute_smd_continuous(x1, x2):
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = np.sqrt((var1 + var2) / 2)
    return abs(mean1 - mean2) / pooled_sd if pooled_sd > 0 else 0

def compute_smd_categorical(x1, x2):
    p1, p2 = np.mean(x1), np.mean(x2)
    pooled = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    return abs(p1 - p2) / np.sqrt(pooled) if pooled > 0 else 0

# ===== 4. 循环检验 =====
for col in common_vars:
    if col in categorical_vars:
        # ---- 分类变量 → 卡方检验 ----
        contingency = pd.crosstab(
            pd.concat([train[col], val[col]], axis=0),
            ["Train"]*len(train) + ["Validation"]*len(val)
        )
        chi2, p, dof, exp = chi2_contingency(contingency)

        smd = compute_smd_categorical(train[col].dropna(), val[col].dropna())
        results.append([col, "Categorical", p, smd])

    elif col in continuous_vars:
        # ---- 连续变量 → Mann-Whitney U 检验 ----
        u_stat, p = stats.mannwhitneyu(
            train[col].dropna(), val[col].dropna(),
            alternative='two-sided'
        )
        smd = compute_smd_continuous(train[col].dropna(), val[col].dropna())
        results.append([col, "Continuous", p, smd])

    else:
        continue

# ===== 5. 保存结果 =====
result_df = pd.DataFrame(results, columns=["Variable", "Type", "p_value", "SMD"])
result_df.to_csv("train_val_comparison_pvalues_smd.csv", index=False)

print("检验完成，结果已保存为 train_val_comparison_pvalues_smd.csv")
