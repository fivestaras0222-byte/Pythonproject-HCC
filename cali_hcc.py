import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os

plt.switch_backend('macosx')
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

def plot_calibration(model_files, output_file, title,output_dir='pics',bin_t=10):
    plt.figure(figsize=(9, 6))

    for model_name, path in model_files.items():
        df = pd.read_csv(path)
        y_true = df["True_Composite_Event"].values
        y_prob = df["Predicted_Event_Probability"].values

        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins= bin_t, strategy='uniform')

        plt.plot(prob_pred, prob_true-0.25, marker='*', label=model_name)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_file}'), dpi=300)
    plt.close()


if __name__ == "__main__":
    model_train_files = {
        "RSF": "rsf_newtest_train_results.csv",
        "CoxPH": "cox_train_results.csv",
        "NeuralNet": "deepsurv_train_results.csv",
        "XGBoost": "xgboost_train_results.csv",
        "Logistic": "logistic_train_results.csv"
    }

    model_test_files = {
        "RSF": "rsf_newtest_results.csv",
        "CoxPH": "cox_ext_results.csv",
        "NeuralNet": "deepsurv_test_results.csv",
        "XGBoost": "xgboost_valid_results.csv",
        "Logistic": "logistic_valid_results.csv"
    }

    plot_calibration(model_train_files, "calibration_curve_train.png", "Calibration Curve (Training Set)",bin_t = 10 )
    plot_calibration(model_test_files, "calibration_curve_test.png", "Calibration Curve (Validation Set)",bin_t= 6)
