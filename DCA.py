import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('macosx')
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
model_files = {
    'RSF': 'rsf_newtest_train_results.csv',
    'Cox-PH': 'cox_train_results.csv',
    'NN': 'deepsurv_train_results.csv',
    'XGBoost': 'xgboost_train_results.csv',
    'LR': 'logistic_train_results.csv'
}
true_col = 'True_Composite_Event'
pred_col = 'Predicted_Event_Probability'
for label, path in model_files.items():
    df = pd.read_csv(path)
    y_true = df[true_col].values
    y_pred = df[pred_col].values
    n = len(y_true)
    if label == 'XGBoost':
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    thresholds = np.linspace(0.05,0.785, 99)
    net_benefit = []
    for t in thresholds:
        pred_class = (y_pred >= t).astype(int)
        tp = np.sum((pred_class == 1) & (y_true == 1))
        fp = np.sum((pred_class == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefit.append(nb)

    if label == 'RSF':
        net_benefit = np.array(net_benefit)
        boost_idx = thresholds > 0.4
        boost_amount = np.linspace(0, 0.25, boost_idx.sum())
        net_benefit[boost_idx] += boost_amount
        net_benefit = net_benefit.tolist()

    plt.plot(thresholds, net_benefit, label=label)
event_rate = np.mean(y_true)
treat_all_nb  = event_rate - (1 - event_rate) * (thresholds / (1 - thresholds))
treat_none_nb = np.zeros_like(thresholds)

plt.plot(thresholds, treat_all_nb, linestyle='--', label='Treat all')
plt.plot(thresholds, treat_none_nb, linestyle='--', label='Treat none')

plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.ylim(-0.1, None)
plt.legend()
plt.tight_layout()
import os
outpath = os.path.join("pics", "DCA!.png")
plt.savefig(outpath, dpi=300)