# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from lifelines import KaplanMeierFitter
# from sklearn.calibration import calibration_curve
# import pandas as pd
# import matplotlib as mpl
# import seaborn as sns
#
# plt.rcParams["font.sans-serif"] = ["Times New Roman"]  # Use standard font
# plt.rcParams["axes.unicode_minus"] = False  # Ensure negative signs display correctly
#
# # 设置 Matplotlib 后端
# plt.switch_backend("macosx")
#
#
# # def plot_feature_correlation(X, output_dir):
# #     """
# #     绘制特征相关性热图，并将其保存到 `output_dir`
# #     """
# #     if X.empty:
# #         print("❌ 错误：X_train 为空，无法绘制特征相关性热图！")
# #         return
# #
# #     correlation_matrix = X.corr()
# #
# #     if correlation_matrix.isnull().all().all():
# #         print("❌ 错误：相关性矩阵为空，可能是所有特征都是非数值型或数据缺失！")
# #         return
# #
# #     plt.figure(figsize=(12, 8))
# #     correlation_matrix = X.corr()
# #
# #     sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# #     plt.title("Feature Correlation Heatmap")
# #     plt.xticks(rotation=90)
# #     plt.yticks(rotation=0)
# #
# #     # 保存热图
# #     heatmap_path = os.path.join(output_dir, "feature_correlation_heatmap.png")
# #     plt.savefig(heatmap_path)
# #     plt.close()
# #     print(f"✅ 特征相关性热图已保存: {heatmap_path}")
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
plt.switch_backend("macosx")
plt.rcParams["font.sans-serif"] = ["Times New Roman"]  # Use standard font
plt.rcParams["axes.unicode_minus"] = False  # Ensure negative signs display correctly

# 加载预测结果数据
df = pd.read_csv("rsf_newtest_results.csv")

# 假设文件中包含 'True_Composite_Event'（真实标签）和 'Predicted_Class'（预测标签）
y_true = df['True_Composite_Event']
y_pred = df['Predicted_Class']

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("RSF",fontsize=16)
plt.colorbar()

# 设置坐标轴标签
classes = np.unique(np.concatenate((np.array(y_true), np.array(y_pred))))
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# 在图中添加数字
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black", size=18)

plt.ylabel("True Label",fontsize=14)
plt.xlabel("Predicted Label", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join("plots_TEST", "COMFUS_MATRIX.png"),dpi=600)
#
# # def plot_visualizations(train_results_csv="rsf_newtest_train_results.csv",
# #                         test_results_csv="rsf_newtest_results.csv",
# #                         output_dir="plots_RSF"):
# #     """
# #     Reads the training and test prediction results CSV and generates:
# #     1. ROC Curve (for both training and test sets)
# #     2. Kaplan-Meier Survival Curve (test set)
# #     3. Calibration Curve (test set)
# #     4. Survival Probability Distribution Histogram (test set)
# #
# #     All plots are saved in the `output_dir` folder instead of being displayed.
# #     """
# #
# #     # 确保输出目录存在
# #     if not os.path.exists(output_dir):
# #         os.makedirs(output_dir)
# #
# #     # ==== 加载训练集和测试集数据 ====
# #     train_results = pd.read_csv(train_results_csv)
# #     test_results = pd.read_csv(test_results_csv)
# #
# #     feature_columns = [col for col in train_results.columns if col not in ["True_Composite_Event",
# #                                                                            "Predicted_Survival_Probability",
# #                                                                            "Predicted_Event_Probability",
# #                                                                            "Predicted_Class"]]
# #     X_train = train_results[feature_columns]
# #
# #     plot_feature_correlation(X_train, output_dir)
# # #
# # #     # ==== 计算训练集 AUC ====
# # #     fpr_train, tpr_train, _ = roc_curve(train_results["True_Composite_Event"], train_results["Predicted_Event_Probability"])
# # #     roc_auc_train = auc(fpr_train, tpr_train)
# # #
# # #     # ==== 计算测试集 AUC ====
# # #     fpr_test, tpr_test, _ = roc_curve(test_results["True_Composite_Event"], test_results["Predicted_Event_Probability"])
# # #     roc_auc_test = auc(fpr_test, tpr_test)
# # #
# # #     # 将 AUC 结果保存到文本文件
# # #     auc_results_file = os.path.join(output_dir, "auc_results.txt")
# # #     with open(auc_results_file, "w") as f:
# # #         f.write(f"Training Set AUC: {roc_auc_train:.4f}\n")
# # #         f.write(f"Test Set AUC: {roc_auc_test:.4f}\n")
# # #
# # #     print(f"✅ Training Set AUC: {roc_auc_train:.4f}")
# # #     print(f"✅ Test Set AUC: {roc_auc_test:.4f}")
# # #     print(f"✅ AUC results saved in: {auc_results_file}")
# # #
# # #     # ==== 训练集 & 测试集 ROC 曲线 ====
# # #     plt.figure(figsize=(10, 5))
# # #     plt.plot(fpr_train, tpr_train, color='green', lw=2, label=f'Training ROC (AUC = {roc_auc_train:.3f})')
# # #     plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'Test ROC (AUC = {roc_auc_test:.3f})')
# # #     plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
# # #     plt.xlabel('False Positive Rate')
# # #     plt.ylabel('True Positive Rate')
# # #     plt.title('ROC Curve (Training & Test Sets)')
# # #     plt.legend(loc="lower right")
# # #     plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
# # #     plt.close()
# # #
# # #     # # ==== Kaplan-Meier 生存曲线（测试集） ====
# # #     # kmf = KaplanMeierFitter()
# # #     # kmf.fit(test_results["Predicted_Survival_Probability"], event_observed=test_results["True_Composite_Event"])
# # #     #
# # #     # plt.figure(figsize=(10, 5))
# # #     # kmf.plot_survival_function()
# # #     # plt.xlabel("Time per 1000 days")
# # #     # plt.ylabel("Survival Probability")
# # #     # plt.title("Kaplan-Meier Survival Curve (Test Set)")
# # #     # plt.savefig(os.path.join(output_dir, "kaplan_meier_curve.png"),dpi=300)
# # #     # plt.close()
# # #
# # #     # ==== 校准曲线（测试集） ====
# # #     prob_true, prob_pred = calibration_curve(test_results["True_Composite_Event"], test_results["Predicted_Event_Probability"], n_bins=10)
# # #
# # #     plt.figure(figsize=(10, 5))
# # #     plt.plot(prob_pred, prob_true, "s-", label="Predicted Calibration Curve")
# # #     plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
# # #     plt.xlabel("Predicted Event Probability")
# # #     plt.ylabel("Observed Event Rate")
# # #     plt.title("Calibration Curve (Test Set)")
# # #     plt.legend()
# # #     plt.savefig(os.path.join(output_dir, "calibration_curve.png"),dpi=300)
# # #     plt.close()
# # #
# # #     # # ==== 生存概率分布直方图（测试集） ====
# # #     # plt.figure(figsize=(10, 5))
# # #     # plt.hist(test_results["Predicted_Survival_Probability"], bins=20, alpha=0.7, color='blue', edgecolor='black')
# # #     # plt.xlabel("Predicted Survival Probability")
# # #     # plt.ylabel("Number of Samples")
# # #     # plt.title("Survival Probability Distribution (Test Set)")
# # #     # plt.savefig(os.path.join(output_dir, "survival_distribution.png"),dpi=300)
# # #     # plt.close()
# # #     #
# # #     # print(f"✅ All plots have been saved in the folder: {output_dir}")
# # #
# # #
# # # if __name__ == '__main__':
# # #     plot_visualizations(train_results_csv="rsf_newtest_train_results.csv",
# # #                         test_results_csv="rsf_newtest_results.csv",
# # #                         output_dir="plots_TEST")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
plt.switch_backend("macosx")
plt.rcParams["font.sans-serif"] = ["Times New Roman"]  # Use standard font
plt.rcParams["axes.unicode_minus"] = False  # Ensure negative signs display correctly
def plot_multiple_confusion_matrices(csv_list, model_names, save_path="plots_TEST/COMFUS_MATRIX_GRID.png"):
    """
    画4个模型的混淆矩阵子图。

    参数:
    - csv_list: 包含4个CSV路径的列表，每个文件需包含'True_Composite_Event' 和 'Predicted_Class'列
    - model_names: 与csv_list长度相同的模型名称列表，用作子图标题
    - save_path: 图像保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()  # 展平为1维，便于循环

    for idx, (csv_file, model_name) in enumerate(zip(csv_list, model_names)):
        df = pd.read_csv(csv_file)
        y_true = df['True_Composite_Event']
        y_pred = df['Predicted_Class']

        cm = confusion_matrix(y_true, y_pred)
        classes = np.unique(np.concatenate((np.array(y_true), np.array(y_pred))))

        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{model_name}", fontsize= 16)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)

        # 添加数值标注
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", size= 18)

        ax.set_ylabel('True Label', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()

csv_files = [
    "cox_ext_results.csv",
    "deepsurv_test_results.csv",
    "xgboost_valid_results.csv",
    "logistic_valid_results.csv"
]

model_names = ["Cox-PH", "NeuralNet", "XGBoost", "LogisticRegression"]

plot_multiple_confusion_matrices(csv_files, model_names)
