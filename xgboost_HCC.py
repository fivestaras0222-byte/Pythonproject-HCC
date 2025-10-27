# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from lifelines.utils import concordance_index
# from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
#
# from sklearn.utils import resample
# from sklearn.metrics import roc_auc_score
# import warnings
#
# def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, ci=0.95, random_state=42):
#     np.random.seed(random_state)
#     aucs = []
#
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#
#     for _ in range(n_bootstraps):
#         # bootstrap抽样
#         indices = np.random.randint(0, len(y_true), len(y_true))
#         if len(np.unique(y_true[indices])) < 2:
#             continue  # 防止只有一个类别
#         score = roc_auc_score(y_true[indices], y_pred[indices])
#         aucs.append(score)
#
#     sorted_scores = np.array(aucs)
#     lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
#     upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
#     mean_auc = np.mean(sorted_scores)
#     return mean_auc, lower, upper
#
#
# def load_data(path):
#         df = pd.read_csv(path)
#
#         # 将所有内容尝试转换为数值（非数值自动变成NaN）
#         for col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#
#         df.fillna(df.median(numeric_only=True), inplace=True)
#         df["composite_event"] = ((df["术后复发（是=1，否=0）"] == 1)).astype(int)
#         return df
#
#
# def prepare_aft_data(df):
#     time = df["时间段"]
#     event = df["composite_event"]
#     # censoring = 1 - event
#     lower_bound = time
#     upper_bound = np.where(event == 1, time, np.inf)
#     return lower_bound, upper_bound
#
#
#
#
# def run_xgb_cox_with_features(
#     data_file="data6_副本.csv",
#     data_new_file="datahx1.csv",
#     features=None,
#     out=1,
#     time_point=730,
#     learning_rate=0.05,
#     max_depth=5,
#     n_estimators=500,
#     min_child_weight=1,
#     gamma=0,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     reg_lambda=1.0,
#     reg_alpha=0.1,
#     tree_method="auto",
#     booster="gbtree"
# ):
#
#     df_train = load_data(data_file)
#     df_valid = load_data(data_new_file)
#
#     if features is None:
#         features = [col for col in df_train.columns if col not in ["时间段", "术后复发（是=1，否=0）", "composite_event"]]
#
#     X_train = df_train[features]
#     X_valid = df_valid[features]
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_valid = scaler.transform(X_valid)
#
#     y_train = df_train["时间段"].values
#     e_train = df_train["composite_event"].values
#
#     y_valid = df_valid["时间段"].values
#     e_valid = df_valid["composite_event"].values
#
#     # 构造 DMatrix，只设置 label 为时间（事件信息通过 weight 模拟处理）
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dtrain.set_float_info("label", y_train)
#     dtrain.set_float_info("weight", e_train)  # 设置删失信息
#
#     dvalid = xgb.DMatrix(X_valid, label=y_valid)
#     dvalid.set_float_info("label", y_valid)
#     dvalid.set_float_info("weight", e_valid)
#
#     params = {
#         "objective": "survival:cox",
#         "eval_metric": "cox-nloglik",
#         "learning_rate": learning_rate,
#         "max_depth": max_depth,
#         "min_child_weight": min_child_weight,
#         "gamma": gamma,
#         "subsample": subsample,
#         "colsample_bytree": colsample_bytree,
#         "lambda": reg_lambda,
#         "alpha": reg_alpha,
#         "tree_method": tree_method,
#         "booster": booster,
#         "n_estimators": n_estimators,
#         "verbosity": 0
#     }
#
#     model = xgb.train(params, dtrain, num_boost_round=300)
#
#     pred_train = model.predict(dtrain)
#     pred_valid = model.predict(dvalid)
#
#     c_index_train = concordance_index(y_train, -pred_train, e_train)
#     c_index_valid = concordance_index(y_valid, -pred_valid, e_valid)
#
#     if out:
#         print(f"✅ 训练集 C-index: {c_index_train:.4f}")
#         print(f"✅ 验证集 C-index: {c_index_valid:.4f}")
#
#
#     # 二分类指标（事件预测）
#     pred_event_train = (pred_train > -np.log(time_point)).astype(int)
#     pred_event_valid = (pred_valid > -np.log(time_point)).astype(int)
#
#     acc_train = accuracy_score(e_train, pred_event_train)
#     acc_valid = accuracy_score(e_valid, pred_event_valid)
#     print(f"✅ Accuracy Train: {acc_train:.4f}")
#     print(f"✅ Accuracy Valid: {acc_valid:.4f}")
#     # 原始风险预测（负值风险越高）
#     risk_train = -pred_train
#     risk_valid = -pred_valid
#
#     # Bootstrap AUC
#     auc_train, ci_lo_train, ci_hi_train = bootstrap_auc_ci(e_train, risk_train)
#     auc_valid, ci_lo_valid, ci_hi_valid = bootstrap_auc_ci(e_valid, risk_valid)
#
#     if out:
#         print(f"✅ AUC Train: {auc_train:.4f} (95% CI: {ci_lo_train:.4f} ~ {ci_hi_train:.4f})")
#         print(f"✅ AUC Valid: {auc_valid:.4f} (95% CI: {ci_lo_valid:.4f} ~ {ci_hi_valid:.4f})")
#
#     return model, acc_train, acc_valid, c_index_train, c_index_valid
#
#
# import matplotlib.pyplot as plt
# from sklearn.calibration import calibration_curve
# from sklearn.metrics import roc_curve, auc
# plt.switch_backend('macosx')
# def plot_roc_and_calibration(y_true, risk_score, output_prefix="cox_model"):
#     """
#     参数：
#     - y_true: 实际事件（1=复发，0=删失）
#     - risk_score: 预测风险（建议为 -xgboost 输出）
#     - output_prefix: 保存文件前缀
#     """
#     # ROC 曲线
#     fpr, tpr, _ = roc_curve(y_true, risk_score)
#     roc_auc = auc(fpr, tpr)
#
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='blue')
#     plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve")
#     plt.legend()
#     plt.savefig(f"{output_prefix}_roc_curve.png", dpi=300)
#     plt.close()
#
#     # 校准曲线
#     score_min, score_max = risk_score.min(), risk_score.max()
#     risk_norm = (risk_score - score_min) / (score_max - score_min + 1e-8)  # 避免除0
#
#     prob_true, prob_pred = calibration_curve(y_true, risk_norm, n_bins=10)
#
#     plt.figure(figsize=(8, 6))
#     plt.plot(prob_pred, prob_true, marker='o', label="Calibration Curve", color='green')
#     plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
#     plt.xlabel("Predicted Risk")
#     plt.ylabel("Observed Event Rate")
#     plt.title("Calibration Curve")
#     plt.legend()
#     plt.savefig(f"{output_prefix}_calibration_curve.png", dpi=300)
#     plt.close()
#
#     print(f"✅ ROC 和校准曲线已保存为: {output_prefix}_*.png")
#
#
# if __name__ == '__main__':
#     features = [
#         'PT',
#         'child分期_A', 'AFP_greater_400',
#                                              '失血量',
#                                              '肿瘤是否巨块型分化_1','AST', 'ALT','WBC','INR','TBIL',
#                                              '年龄__y',
#                                              '性别_1', 'ALBI', '包膜是否受侵犯_未浸及', '肿瘤直径',
#                                              'ALB', '是否合并肝硬化_1',
#                                              '是否大范围切除_1',
#                                              '是否合并肝炎_1', '肿瘤MVI_M0',
#                                              'AFP_less_400'
#         ]
#     model, acc_train, acc_valid, c_index_train, c_index_valid = run_xgb_cox_with_features(features=features, out=1)
#
#     # 再次载入数据以绘图
#     df_valid = load_data("datahx1.csv")
#     X_valid = StandardScaler().fit_transform(df_valid[features])
#     y_valid = df_valid["术后复发（是=1，否=0）"].values
#     dvalid = xgb.DMatrix(X_valid)
#     pred_valid = model.predict(dvalid)
#     risk_valid = -pred_valid  # 风险分数越高，复发概率越大
#
#     # 绘制曲线
#     plot_roc_and_calibration(y_valid, risk_valid, output_prefix="cox_valid")
#     # 保存验证集预测结果为 CSV，用于后续比较绘图
#     pd.DataFrame({
#         "True_Composite_Event": y_valid,
#         "Predicted_Event_Probability": risk_valid  # 这里是负的 log 风险，数值越大表示风险越高
#     }).to_csv("xgboost_valid_results.csv", index=False)
#     print("✅ XGBoost 验证集预测结果已保存为 xgboost_valid_results.csv")
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss,confusion_matrix

from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
import warnings
def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:  # 避免单类别
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    lower = np.percentile(scores, (1-ci)/2*100)
    upper = np.percentile(scores, (1+ci)/2*100)
    mean = np.mean(scores)
    return mean, lower, upper

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, ci=0.95, random_state=42):
    np.random.seed(random_state)
    aucs = []

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(n_bootstraps):
        # bootstrap抽样
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue  # 防止只有一个类别
        score = roc_auc_score(y_true[indices], y_pred[indices])
        aucs.append(score)

    sorted_scores = np.array(aucs)
    lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
    mean_auc = np.mean(sorted_scores)
    return mean_auc, lower, upper


def load_data(path):
        df = pd.read_csv(path)

        # 将所有内容尝试转换为数值（非数值自动变成NaN）
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.fillna(df.median(numeric_only=True), inplace=True)
        df['composite_event'] = ((df['术后复发（是=1，否=0）'] == 1) | (df['target'] == 1)).astype(int)
        return df


def prepare_aft_data(df):
    time = df["时间段"]
    event = df["composite_event"]
    # censoring = 1 - event
    lower_bound = time
    upper_bound = np.where(event == 1, time, np.inf)
    return lower_bound, upper_bound




def run_xgb_aft_with_features(
    data_file="data6_副本.csv",
    data_new_file="datahx1.csv",
    features=None,
    out=1,
    time_point=730,
    learning_rate=0.01,
    max_depth=3,
    n_estimators=500,
    min_child_weight=2,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    reg_alpha=1,
    aft_loss_distribution="normal",
    aft_loss_distribution_scale=1.0,
    tree_method="auto",
    booster="gbtree"
):

    df_train = load_data(data_file)
    df_valid = load_data(data_new_file)

    if features is None:
        features = [col for col in df_train.columns if col not in ["时间段", "术后复发（是=1，否=0）", "composite_event"]]

    X_train = df_train[features]
    X_valid = df_valid[features]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    y_train_lower, y_train_upper = prepare_aft_data(df_train)
    y_valid_lower, y_valid_upper = prepare_aft_data(df_valid)

    dtrain = xgb.DMatrix(X_train)
    dtrain.set_float_info("label_lower_bound", y_train_lower)
    dtrain.set_float_info("label_upper_bound", y_train_upper)

    dvalid = xgb.DMatrix(X_valid)
    dvalid.set_float_info("label_lower_bound", y_valid_lower)
    dvalid.set_float_info("label_upper_bound", y_valid_upper)

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": aft_loss_distribution,
        "aft_loss_distribution_scale": aft_loss_distribution_scale,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "tree_method": tree_method,
        "booster": booster,
        "verbosity": 0
    }

    model = xgb.train(params, dtrain, num_boost_round=n_estimators)
    model.set_attr(aft_sigma=str(params["aft_loss_distribution_scale"]))
    pred_train = model.predict(dtrain)
    pred_valid = model.predict(dvalid)

    e_train = df_train["composite_event"].values
    e_valid = df_valid["composite_event"].values
    t_train = df_train["时间段"].values
    t_valid = df_valid["时间段"].values
    from lifelines.utils import concordance_index
    c_index_train = concordance_index(t_train, pred_train, e_train)
    c_index_valid = concordance_index(t_valid, pred_valid, e_valid)

    if out:
        print(f"✅ AFT 训练集 C-index: {c_index_train:.4f}")
        print(f"✅ AFT 验证集 C-index: {c_index_valid:.4f}")
    # 二分类判断：是否在 time_point 之前复发
    pred_event_train = (pred_train < time_point).astype(int)  # 预测时间短 → 风险高 → 预测复发
    pred_event_valid = (pred_valid < time_point).astype(int)

    acc_train = accuracy_score(e_train, pred_event_train)
    acc_valid = accuracy_score(e_valid, pred_event_valid)

    if out:
        print(f"✅ Accuracy Train: {acc_train:.4f}")
        print(f"✅ Accuracy Valid: {acc_valid:.4f}")


    # risk = -pred，越小风险越大
    auc_train, ci_lo_train, ci_hi_train = bootstrap_auc_ci(e_train, -pred_train)
    auc_valid, ci_lo_valid, ci_hi_valid = bootstrap_auc_ci(e_valid, -pred_valid)
    auc_train1 = roc_auc_score(e_train, -pred_train)
    auc_valid1 = roc_auc_score(e_valid, -pred_valid)
    cm1 = confusion_matrix(e_train, pred_event_train)
    cm = confusion_matrix(e_valid, pred_event_valid)

    from lifelines.utils import concordance_index
    from sklearn.metrics import brier_score_loss

    # C-index
    cidx_train_mean, cidx_train_lo, cidx_train_hi = bootstrap_metric_ci(
        t_train, pred_train, lambda y_true, y_pred: concordance_index(y_true, y_pred, e_train)
    )
    cidx_valid_mean, cidx_valid_lo, cidx_valid_hi = bootstrap_metric_ci(
        t_valid, pred_valid, lambda y_true, y_pred: concordance_index(y_true, y_pred, e_valid)
    )


    if out:
        print(f"✅ AUC Train: {auc_train:.4f} (95% CI: {ci_lo_train:.4f} ~ {ci_hi_train:.4f})")
        print(f"✅ AUC Valid: {auc_valid:.4f} (95% CI: {ci_lo_valid:.4f} ~ {ci_hi_valid:.4f})")
        print(f"✅ C-index Train: {cidx_train_mean:.4f} (95% CI: {cidx_train_lo:.4f}-{cidx_train_hi:.4f})")
        print(f"✅ C-index Valid: {cidx_valid_mean:.4f} (95% CI: {cidx_valid_lo:.4f}-{cidx_valid_hi:.4f})")

        print(cm1)
        print(cm)
        print(auc_train1,auc_valid1)
    return scaler, model, acc_train, acc_valid, c_index_train, c_index_valid,e_train,e_valid,pred_train,pred_valid,pred_event_train,pred_event_valid



import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
plt.switch_backend('macosx')
def plot_roc_and_calibration(y_true, risk_score, output_prefix="cox_model"):
    """
    参数：
    - y_true: 实际事件（1=复发，0=删失）
    - risk_score: 预测风险（建议为 -xgboost 输出）
    - output_prefix: 保存文件前缀
    """
    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, risk_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{output_prefix}_roc_curve.png", dpi=300)
    plt.close()

    # 校准曲线
    score_min, score_max = risk_score.min(), risk_score.max()
    risk_norm = (risk_score - score_min) / (score_max - score_min + 1e-8)  # 避免除0

    prob_true, prob_pred = calibration_curve(y_true, risk_norm, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label="Calibration Curve", color='green')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    plt.xlabel("Predicted Risk")
    plt.ylabel("Observed Event Rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(f"{output_prefix}_calibration_curve.png", dpi=300)
    plt.close()

    print(f"✅ ROC 和校准曲线已保存为: {output_prefix}_*.png")


if __name__ == '__main__':
    features = [
        'PT',
        'child分期_A', 'AFP_greater_400',
                                             '失血量',
                                             '肿瘤是否巨块型分化_1','AST', 'ALT','WBC','INR','TBIL',
                                             '年龄__y',
                                             '性别_1', 'ALBI', '包膜是否受侵犯_未浸及', '肿瘤直径',
                                             'ALB', '是否合并肝硬化_1',
                                             '是否大范围切除_1',
                                             '是否合并肝炎_1', '肿瘤MVI_M0',
                                             'AFP_less_400'
        ]
    scaler, model, acc_train, acc_valid, c_index_train, c_index_valid,e_train,e_valid,pred_train,pred_valid,pred_event_train,pred_event_valid = run_xgb_aft_with_features(features=features, out=1)
    model.save_model("aft_model.ubj")
    import joblib
    joblib.dump(scaler, "xgb_scaler.pkl")


    #
    # # 绘制曲线
    # plot_roc_and_calibration(y_valid, risk_valid, output_prefix="cox_valid")
    # 保存验证集预测结果为 CSV，用于后续比较绘图
    pd.DataFrame({
        "True_Composite_Event": e_valid,
        "Predicted_Event_Probability": -pred_valid,  # 这里是负的 log 风险，数值越大表示风险越高
        "Predicted_Class": pred_event_valid
    }).to_csv("xgboost_valid_results.csv", index=False)
    print("✅ XGBoost 验证集预测结果已保存为 xgboost_valid_results.csv")


    # 保存为 CSV 文件
    pd.DataFrame({
        "True_Composite_Event": e_train,
        "Predicted_Event_Probability": -pred_train,
        "Predicted_Class": pred_event_train
    }).to_csv("xgboost_train_results.csv", index=False)
    print("✅ XGBoost 训练集预测结果已保存为 xgboost_train_results.csv")
    # # 再次载入数据以绘图
    df_valid = load_data("datahx1.csv")
    X_valid = StandardScaler().fit_transform(df_valid[features])
    y_valid = df_valid["术后复发（是=1，否=0）"].values
    dvalid = xgb.DMatrix(X_valid)
    pred_valid = model.predict(dvalid)
    risk_valid = -pred_valid  # 风险分数越高，复发概率越大
    # # 保存训练集预测结果为 CSV
    df_train = load_data("data6_副本.csv")  # 使用你的训练集文件名
    X_train = StandardScaler().fit_transform(df_train[features])
    y_train = df_train["术后复发（是=1，否=0）"].values
    dtrain = xgb.DMatrix(X_train)
    pred_train = model.predict(dtrain)
    risk_train = -pred_train  # 风险越大，复发概率越高

    from sklearn.metrics import brier_score_loss

    # 预测概率归一化（因为 risk 是未校准的负预测时间）
    risk_valid_norm = (risk_valid - risk_valid.min()) / (risk_valid.max() - risk_valid.min() + 1e-8)
    risk_train_norm = (risk_train - risk_train.min()) / (risk_train.max() - risk_train.min() + 1e-8)

    # Brier Score
    brier_valid = brier_score_loss(e_valid, risk_valid_norm)
    brier_train = brier_score_loss(e_train, risk_train_norm)

    print(f"✅ Brier Score Train: {brier_train:.4f}")
    print(f"✅ Brier Score Valid: {brier_valid:.4f}")
    # Brier Score 及 CI
    brier_train_mean, brier_train_lo, brier_train_hi = bootstrap_metric_ci(e_train, risk_train_norm, brier_score_loss)
    brier_valid_mean, brier_valid_lo, brier_valid_hi = bootstrap_metric_ci(e_valid, risk_valid_norm, brier_score_loss)

    print(f"✅ Brier Score Train: {brier_train_mean:.4f} (95% CI: {brier_train_lo:.4f} ~ {brier_train_hi:.4f})")
    print(f"✅ Brier Score Valid: {brier_valid_mean:.4f} (95% CI: {brier_valid_lo:.4f} ~ {brier_valid_hi:.4f})")
    print("sigma:", model.attr("aft_sigma"))


