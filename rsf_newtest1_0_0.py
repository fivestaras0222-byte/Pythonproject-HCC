import re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import brier_score_loss

available_features = [
    '性别_1', '是否合并肝炎_1', '是否合并肝硬化_1', '包膜是否受侵犯_未浸及', '肿瘤是否巨块型分化_1',
    '是否开腹手术（1=开腹，无=腹腔镜手术）_1', '肿瘤MVI_M0', '是否大范围切除_1', 'AFP_less_400', 'AFP_greater_400', '年龄__y',
    'ALT', 'AST', 'TBIL', 'R-GGT', 'ALP', 'ALB', 'PT', 'INR', 'WBC', 'PLT', '肿瘤直径', '失血量'
]

def downsample_positive(df_valid, event_col, target_rate, random_state=42):
    rng = np.random.default_rng(random_state)

    neg_df = df_valid[df_valid[event_col] == 0]
    pos_df = df_valid[df_valid[event_col] == 1]

    n_neg = len(neg_df)
    n_pos_target = int(round(target_rate / (1 - target_rate) * n_neg))
    n_pos = min(len(pos_df), n_pos_target)

    pos_sampled = pos_df.sample(n=n_pos, random_state=random_state)
    new_valid = pd.concat([neg_df, pos_sampled]).sample(frac=1, random_state=random_state)
    return new_valid


def stratified_subsample(df_valid, event_col, target_rate, random_state=42):
    rng = np.random.default_rng(random_state)
    pos_df = df_valid[df_valid[event_col] == 1]
    neg_df = df_valid[df_valid[event_col] == 0]
    n_total = len(df_valid)
    n_pos_target = int(round(target_rate * n_total))
    n_neg_target = n_total - n_pos_target
    n_pos = min(len(pos_df), n_pos_target)
    n_neg = min(len(neg_df), n_neg_target)
    if n_pos < n_pos_target:
        n_neg = min(len(neg_df), n_total - n_pos)
    if n_neg < n_neg_target:
        n_pos = min(len(pos_df), n_total - n_neg)

    sampled_pos = pos_df.sample(n=n_pos, random_state=random_state)
    sampled_neg = neg_df.sample(n=n_neg, random_state=random_state)
    new_valid = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=random_state)  # 打乱
    return new_valid


def bootstrap_auc_ci(y_true, y_pred_proba, n_bootstraps=1000, alpha=0.95, random_state=42):
    rng = np.random.default_rng(random_state)
    auc_values = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_values.append(roc_auc_score(y_true[idx], y_pred_proba[idx]))

    lower = np.percentile(auc_values, (1 - alpha) / 2 * 100)
    upper = np.percentile(auc_values, (1 + alpha) / 2 * 100)
    return float(np.mean(auc_values)), float(lower), float(upper)


def bootstrap_brier_ci(y_true, y_pred_proba, n_bootstraps=1000, alpha=0.95, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    briers = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        briers.append(brier_score_loss(y_true[idx], y_pred_proba[idx]))
    lower = np.percentile(briers, (1 - alpha) / 2 * 100)
    upper = np.percentile(briers, (1 + alpha) / 2 * 100)
    return float(np.mean(briers)), float(lower), float(upper)


def bootstrap_cindex_ci(y_struct, times, risk_scores, n_bootstraps=1000, alpha=0.95, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(times)
    c_vals = []
    try:
        c0 = concordance_index_censored(y_struct["event"], times, risk_scores)[0]
        c_vals.append(c0)
    except Exception:
        pass

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        try:
            c = concordance_index_censored(y_struct["event"][idx], times[idx], risk_scores[idx])[0]
            if np.isfinite(c):
                c_vals.append(c)
        except Exception:
            continue

    if len(c_vals) == 0:
        return np.nan, np.nan, np.nan

    lower = np.percentile(c_vals, (1 - alpha) / 2 * 100)
    upper = np.percentile(c_vals, (1 + alpha) / 2 * 100)
    return float(np.mean(c_vals)), float(lower), float(upper)


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


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.map(extract_numeric)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    df["composite_event"] = ((df["术后复发（是=1，否=0）"] == 1) | (df["target"] == 1)).astype(int)
    return df


def prepare_survival_data(df):
    time = df["时间段"].astype(float)
    y_survival = np.array(
        [(bool(ev), t) for ev, t in zip(df["composite_event"], time)],
        dtype=[("event", "?"), ("time", "<f8")]
    )
    return y_survival


def train_rsf_model(X_train, y_train, random_state=42):
    rsf = RandomSurvivalForest(
        n_estimators=1800,
        max_features='sqrt',
        min_samples_split=2,
        min_samples_leaf=7,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    rsf.fit(X_train, y_train)
    return rsf


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return specificity, sensitivity, ppv, npv


def check_class_balance(y_true):
    """
    检查 y_true 中是否只有一种类别。
    如果只有一种类别，返回 True，否则返回 False。
    """
    y_true_series = pd.Series(y_true)
    unique_classes = y_true_series.unique()
    return len(unique_classes) == 1

def run_rsf_with_imported_features(
    data_file="data6_副本.csv", data_new_file="datahx1.csv",
    test_size=0, random_state=42, time_point=730,
    features=available_features, out=0
):
    threshold = 0.537
    data600_df = load_and_preprocess_data(data_file)
    data_new_df = load_and_preprocess_data(data_new_file)
    p_train = data600_df['composite_event'].mean()
    print('ptrain：', p_train)
    data_new_df = downsample_positive(
        data_new_df,
        event_col='composite_event',
        target_rate=p_train,
        random_state=random_state
    )

    y_survival1 = prepare_survival_data(data600_df)
    y_survival2 = prepare_survival_data(data_new_df)

    X1 = data600_df.drop(columns=["时间段", "target", "术后复发（是=1，否=0）", "composite_event"], errors="ignore")
    X2 = data_new_df.drop(columns=["时间段", "target", "术后复发（是=1，否=0）", "composite_event"], errors="ignore")
    print(features)
    X1 = X1[features]
    X2 = X2[features]
    y1 = data600_df['composite_event'].astype(int).to_numpy()
    y2 = data_new_df['composite_event'].astype(int).to_numpy()
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X1, y1, test_size=test_size, random_state=random_state, stratify=y1
        )
        train_idx = X_train.index.tolist()
        test_idx = X_test.index.tolist()
        y_surv_train = y_survival1[train_idx]
        y_surv_test = y_survival1[test_idx]
    else:
        X_train = X1
        y_train = y1
        train_idx = X_train.index.tolist()
        y_surv_train = y_survival1[train_idx]

    best_rsf = train_rsf_model(X_train, y_surv_train, random_state=random_state)
    y_pred_surv_funcs_train = best_rsf.predict_survival_function(X_train)
    y_pred_surv_prob_train = np.array([fn(time_point) for fn in y_pred_surv_funcs_train])
    pred_event_prob_train = 1 - y_pred_surv_prob_train
    pred_binary_train = (y_pred_surv_prob_train < threshold).astype(int)

    accuracy_train = accuracy_score(y_train, pred_binary_train)
    c_index_train = concordance_index_censored(
        y_surv_train["event"], data600_df.loc[train_idx, "时间段"].to_numpy(dtype=float), pred_event_prob_train
    )[0]
    c_train_mean, c_train_l, c_train_u = bootstrap_cindex_ci(
        y_surv_train,
        data600_df.loc[train_idx, "时间段"].to_numpy(dtype=float),
        pred_event_prob_train,
        n_bootstraps=1000,
        alpha=0.95,
        random_state=random_state,
    )
    brier_train = brier_score_loss(y_train, pred_event_prob_train)
    brier_train_mean, brier_train_l, brier_train_u = bootstrap_brier_ci(
        y_train, pred_event_prob_train, n_bootstraps=1000, alpha=0.95, random_state=random_state
    )

    print(f"✅ Training C-index: {c_index_train:.4f} (Boot mean {c_train_mean:.4f}, 95% CI {c_train_l:.4f}-{c_train_u:.4f})")
    if out:
        print(f"Training Accuracy: {accuracy_train:.4f}")
        print(f"✅ 训练集 Brier score: {brier_train:.4f} (Boot mean {brier_train_mean:.4f}, 95% CI {brier_train_l:.4f}-{brier_train_u:.4f})")

        train_auc_mean, train_auc_lower, train_auc_upper = bootstrap_auc_ci(
            y_train, pred_event_prob_train, n_bootstraps=1000, alpha=0.95, random_state=random_state
        )
        print(f"✅ 训练集 AUC: {train_auc_mean:.4f} (95% CI: {train_auc_lower:.4f} - {train_auc_upper:.4f})")

        specificity_train, sensitivity_train, ppv_train, npv_train = calculate_metrics(y_train, pred_binary_train)
        print(f"✅ Training - Specificity: {specificity_train:.4f}, Sensitivity: {sensitivity_train:.4f}, PPV: {ppv_train:.4f}, NPV: {npv_train:.4f}")
        print("Class distribution in the training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        print(dict(zip(unique, counts)))
        train_results = pd.DataFrame({
            "True_Composite_Event": y_train,
            "Predicted_Survival_Probability": y_pred_surv_prob_train,
            "Predicted_Event_Probability": pred_event_prob_train,
            "Predicted_Class": pred_binary_train
        })
        train_results.to_csv("rsf_newtest_train_results.csv", index=False)
    y_surv_valid = prepare_survival_data(data_new_df)
    X2 = data_new_df[features]
    valid_idx = X2.index.tolist()

    y_pred_surv_funcs = best_rsf.predict_survival_function(X2)
    y_pred_surv_prob = np.array([fn(time_point) for fn in y_pred_surv_funcs])
    pred_event_prob = 1 - y_pred_surv_prob
    pred_binary = (y_pred_surv_prob < threshold).astype(int)

    accuracy = accuracy_score(y2, pred_binary)

    c_index = concordance_index_censored(
        y_surv_valid["event"], data_new_df.loc[valid_idx, "时间段"].to_numpy(dtype=float), pred_event_prob
    )[0]
    c_valid_mean, c_valid_l, c_valid_u = bootstrap_cindex_ci(
        y_surv_valid,
        data_new_df.loc[valid_idx, "时间段"].to_numpy(dtype=float),
        pred_event_prob,
        n_bootstraps=1000,
        alpha=0.95,
        random_state=random_state,
    )

    print(f"✅ External Validation C-index: {c_index:.4f} (Boot mean {c_valid_mean:.4f}, 95% CI {c_valid_l:.4f}-{c_valid_u:.4f})")

    cm = confusion_matrix(y2, pred_binary)
    if out:
        print(f"External Validation Accuracy: {accuracy:.4f}")
        print(cm)

        valid_auc_mean, valid_auc_lower, valid_auc_upper = bootstrap_auc_ci(
            y2, pred_event_prob, n_bootstraps=1000, alpha=0.95, random_state=random_state
        )
        valid_auc = roc_auc_score(y2, pred_event_prob)
        print(valid_auc)
        print(f"✅ valid集 AUC: {valid_auc_mean:.4f} (95% CI: {valid_auc_lower:.4f} - {valid_auc_upper:.4f})")

        brier_valid = brier_score_loss(y2, pred_event_prob)
        brier_valid_mean, brier_valid_l, brier_valid_u = bootstrap_brier_ci(
            y2, pred_event_prob, n_bootstraps=1000, alpha=0.95, random_state=random_state
        )
        print(f"✅ valid集 Brier score: {brier_valid:.4f} (Boot mean {brier_valid_mean:.4f}, 95% CI {brier_valid_l:.4f}-{brier_valid_u:.4f})")

        specificity_test, sensitivity_test, ppv_test, npv_test = calculate_metrics(y2, pred_binary)
        print(f"✅ External Validation - Specificity: {specificity_test:.4f}, Sensitivity: {sensitivity_test:.4f}, PPV: {ppv_test:.4f}, NPV: {npv_test:.4f}")
        print("Class distribution in the validation set:")
        unique, counts = np.unique(y2, return_counts=True)
        print(dict(zip(unique, counts)))

        from evaluate_metrics import compute_metrics
        train_metrics = compute_metrics(y_train, pred_binary_train)
        test_metrics = compute_metrics(y2, pred_binary)

        time_point_val = float(time_point)
        actual_survival_rate_train = np.mean(data600_df.loc[train_idx, "时间段"].to_numpy(dtype=float) >= time_point_val)
        actual_survival_rate_valid = np.mean(data_new_df.loc[valid_idx, "时间段"].to_numpy(dtype=float) >= time_point_val)
        predicted_survival_rate_train = np.mean(pred_binary_train == 0)
        predicted_survival_rate_valid = np.mean(pred_binary == 0)

        # print(f"✅ 训练集 {time_point} 天后实际生存率: {actual_survival_rate_train:.4f}")
        # print(f"✅ 训练集 {time_point} 天后预测生存率: {predicted_survival_rate_train:.4f}")
        # print(f"✅ 测试集 {time_point} 天后实际生存率: {actual_survival_rate_valid:.4f}")
        # print(f"✅ 测试集 {time_point} 天后预测生存率: {predicted_survival_rate_valid:.4f}")

        test_results = pd.DataFrame({
            "True_Composite_Event": y2,
            "Predicted_Survival_Probability": y_pred_surv_prob,
            "Predicted_Event_Probability": pred_event_prob,
            "Predicted_Class": pred_binary
        })
        test_results.to_csv("rsf_newtest_results.csv", index=False)
        print("✅ 测试集预测结果已保存: rsf_newtest_results.csv")
        joblib.dump(best_rsf, "newbest_rsf_model.pkl")
        print("✅ RSF模型已保存为 newbest_rsf_model.pkl")

    return accuracy_train, accuracy


if __name__ == '__main__':
    run_rsf_with_imported_features(
        features=['PT', 'child分期_A', 'AFP_greater_400',
                  '失血量',
                  '肿瘤是否巨块型分化_1','AST', 'ALT','WBC','INR','TBIL',
                  '年龄__y',
                  '性别_1', 'ALBI', '包膜是否受侵犯_未浸及', '肿瘤直径',
                  'ALB', '是否合并肝硬化_1',
                  '是否大范围切除_1',
                  '是否合并肝炎_1', '肿瘤MVI_M0',
                  'AFP_less_400'],
        out=1
    )
