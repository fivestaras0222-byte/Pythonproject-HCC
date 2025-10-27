import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, roc_auc_score
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.switch_backend('macosx')

def plot_combined_roc_curves(model_files, output_path="combined_roc_curve.png",output_dir = 'pics'):

    plt.figure(figsize=(10, 7))

    for model_name, (file_path, flip) in model_files.items():
        df = pd.read_csv(file_path)
        y_true = df["True_Composite_Event"]
        y_prob = df["Predicted_Event_Probability"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        if flip:
            # 沿 y=x 对角线翻折
            fpr, tpr = 1 - fpr, 1 - tpr
            roc_auc = 1 - roc_auc
            label = f"{model_name} (AUC = {roc_auc:.4f})"
        else:
            label = f"{model_name} (AUC = {roc_auc:.4f})"

        plt.plot(fpr, tpr, lw=2, label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{output_path}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{output_path}'), dpi=300)
    plt.close()
    return roc_auc
if __name__ == "__main__":
    model_result_files = {
        "RSF":       ("rsf_newtest_train_results.csv", False),
        "CoxPH":     ("cox_train_results.csv", False),
        "NeuralNet": ("deepsurv_train_results.csv", False),
        "XGBoost":   ("xgboost_train_results.csv", False),
        "Logistic Regression": ("logistic_train_results.csv",False)
    }
    plot_combined_roc_curves(model_result_files, output_path="Training Set ROC Curves Comparison.png")
    model_resulttest_files = {
        "RSF": ("rsf_newtest_results.csv", False),
        "CoxPH": ("cox_ext_results.csv", False),
        "NeuralNet": ("deepsurv_test_results.csv", False),
        "XGBoost": ("xgboost_valid_results.csv", False),
        "Logistic Regression": ("logistic_valid_results.csv", False)
    }
    plot_combined_roc_curves(model_resulttest_files, output_path="Validation Set ROC Curves Comparison.png")
