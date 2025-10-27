import re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, brier_score_loss
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored

available_features = ['PT', 'child分期_A', 'AFP_greater_400',
                                             '失血量',
                                             '肿瘤是否巨块型分化_1','AST', 'ALT','WBC','INR','TBIL',
                                             '年龄__y',
                                             '性别_1', 'ALBI', '包膜是否受侵犯_未浸及', '肿瘤直径',
                                             'ALB', '是否合并肝硬化_1',
                                             '是否大范围切除_1',
                                             '是否合并肝炎_1', '肿瘤MVI_M0',
                                             'AFP_less_400'
]

def bootstrap_cindex_ci(y_surv, risk_scores, n_bootstraps=1000, alpha=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_surv)
    cidxs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        y_sample = y_surv[idx]
        risk_sample = risk_scores[idx]
        try:
            cidx = concordance_index_censored(y_sample['event'], y_sample['time'], risk_sample)[0]
            cidxs.append(cidx)
        except:
            continue
    lower = np.percentile(cidxs, (1-alpha)/2*100)
    upper = np.percentile(cidxs, (1+alpha)/2*100)
    return np.mean(cidxs), lower, upper

def bootstrap_brier_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    briers = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        try:
            b = brier_score_loss(y_true[idx], y_pred[idx])
            briers.append(b)
        except:
            continue
    lower = np.percentile(briers, (1-alpha)/2*100)
    upper = np.percentile(briers, (1+alpha)/2*100)
    return np.mean(briers), lower, upper
def sample_external_to_match_training(df_ext, df_train, event_column='composite_event', random_state=42):
    train_event_rate = df_train[event_column].mean()
    df_event = df_ext[df_ext[event_column] == 1]
    df_nonevent = df_ext[df_ext[event_column] == 0]
    max_total = min(len(df_event) / train_event_rate, len(df_nonevent) / (1 - train_event_rate))
    n_total = int(max_total)
    n_event = int(n_total * train_event_rate)
    n_nonevent = n_total - n_event
    df_event_sampled = df_event.sample(n=n_event, random_state=random_state)
    df_nonevent_sampled = df_nonevent.sample(n=n_nonevent, random_state=random_state)
    df_sampled = pd.concat([df_event_sampled, df_nonevent_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df_sampled

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
    df['composite_event'] = ((df['术后复发（是=1，否=0）'] == 1) | (df['target'] == 1)).astype(int)
    return df

def prepare_survival_data(df):
    return np.array(
        [(bool(e), t) for e, t in zip(df['composite_event'], df['时间段'])],
        dtype=[('event', '?'), ('time', '<f8')]
    )

def train_cox_model(X_train, y_surv_train, time_col='time', event_col='event', penalizer=0,l1_ratio=1):
    data = X_train.copy()
    data[event_col] = y_surv_train['event']
    data[time_col]  = y_surv_train['time']
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(data, duration_col=time_col, event_col=event_col)
    summary_df = cph.summary.reset_index()
    idx_col = summary_df.columns[0]
    summary_df = summary_df.rename(columns={
        idx_col: "Variable", "coef": "Coef", "exp(coef)": "HR",
        "coef lower 95%": "CI_lower", "coef upper 95%": "CI_upper", "p": "p_value"
    })
    summary_df = summary_df[["Variable", "Coef", "HR", "CI_lower", "CI_upper", "p_value"]]
    summary_df.to_csv("cox_variable_summary.csv", index=False, encoding="utf-8-sig")
    return cph

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    aucs = []
    n = len(y_true)
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
    return np.mean(aucs), lower, upper

def run_cox_with_external_validation(train_file="data6_副本.csv",
                                     ext_file="datahx1.csv",
                                     features=available_features,
                                     test_size=0.0,
                                     random_state=42,
                                     time_point=730,
                                     threshold=0.606,
                                     out=1):
    df_train = load_and_preprocess_data(train_file)
    df_ext   = load_and_preprocess_data(ext_file)
    y_train_surv = prepare_survival_data(df_train)
    y_ext_surv   = prepare_survival_data(df_ext)
    X_train_full = df_train[features]
    X_ext        = df_ext[features]
    y_train_ev   = df_train['composite_event']
    y_ext_ev     = df_ext['composite_event']

    if test_size > 0:
        X_train, X_val, y_train_ev, y_val_ev, surv_train, surv_val = train_test_split(
            X_train_full, y_train_ev, y_train_surv, test_size=test_size,
            random_state=random_state, stratify=y_train_ev
        )
    else:
        X_train, y_train_ev, surv_train = X_train_full, y_train_ev, y_train_surv

    cph = train_cox_model(X_train, surv_train, time_col='time', event_col='event')
    surv_func_train = cph.predict_survival_function(X_train, times=[time_point]).values.flatten()
    prob_event_train = 1 - surv_func_train
    pred_class_train = (surv_func_train < threshold).astype(int)
    acc_train = accuracy_score(y_train_ev, pred_class_train)
    cidx_train = concordance_index_censored(surv_train['event'], df_train.loc[X_train.index, '时间段'], prob_event_train)[0]
    print(f"Training Accuracy: {acc_train:.4f}, C-index: {cidx_train:.4f}")
    cm1 = confusion_matrix(y_train_ev, pred_class_train)
    if out:
        brier_train = brier_score_loss(y_train_ev, prob_event_train)
        cidx_train_mean, cidx_train_lower, cidx_train_upper = bootstrap_cindex_ci(surv_train, prob_event_train)
        brier_train_mean, brier_train_lower, brier_train_upper = bootstrap_brier_ci(y_train_ev.values, prob_event_train)
        auc_train, l1, u1 = bootstrap_auc_ci(y_train_ev.values, prob_event_train)
        print(f"  Brier: {brier_train:.4f}, AUC: {auc_train:.4f} ({l1:.4f}-{u1:.4f})")
        print(
            f"Training Accuracy: {acc_train:.4f}, C-index: {cidx_train:.4f} (95% CI: {cidx_train_lower:.4f}-{cidx_train_upper:.4f})")
        print(
            f"  Brier: {brier_train_mean:.4f} (95% CI: {brier_train_lower:.4f}-{brier_train_upper:.4f}))")
        print(cm1)
    surv_func_ext = cph.predict_survival_function(X_ext, times=[time_point]).values.flatten()
    prob_event_ext = 1 - surv_func_ext
    pred_class_ext = (surv_func_ext < threshold).astype(int)
    acc_ext = accuracy_score(y_ext_ev, pred_class_ext)
    cidx_ext = concordance_index_censored(y_ext_surv['event'], df_ext['时间段'], prob_event_ext)[0]
    print(f"External Accuracy: {acc_ext:.4f}, C-index: {cidx_ext:.4f}")
    cm = confusion_matrix(y_ext_ev, pred_class_ext)
    if out:
        brier_ext = brier_score_loss(y_ext_ev, prob_event_ext)
        auc_ext, l2, u2 = bootstrap_auc_ci(y_ext_ev.values, prob_event_ext)
        cidx_ext_mean, cidx_ext_lower, cidx_ext_upper = bootstrap_cindex_ci(y_ext_surv, prob_event_ext)
        brier_ext_mean, brier_ext_lower, brier_ext_upper = bootstrap_brier_ci(y_ext_ev.values, prob_event_ext)
        print(f"  Brier: {brier_ext:.4f}, AUC: {auc_ext:.4f} ({l2:.4f}-{u2:.4f})")
        print(
            f"External Accuracy: {acc_ext:.4f}, C-index: {cidx_ext:.4f} (95% CI: {cidx_ext_lower:.4f}-{cidx_ext_upper:.4f})")
        print(
            f"  Brier: {brier_ext_mean:.4f} (95% CI: {brier_ext_lower:.4f}-{brier_ext_upper:.4f})")

        print(cm)
    train_res = pd.DataFrame({
        'True_Composite_Event': y_train_ev,
        'Predicted_Event_Probability': prob_event_train,
        'Predicted_Class': pred_class_train
    })
    train_res.to_csv("cox_train_results.csv", index=False)
    ext_res = pd.DataFrame({
        'True_Composite_Event': y_ext_ev,
        'Predicted_Event_Probability': prob_event_ext,
        'Predicted_Class': pred_class_ext
    })
    ext_res.to_csv("cox_ext_results.csv", index=False)
    print("✅ 已保存预测结果：cox_train_results.csv, cox_ext_results.csv")
    joblib.dump(cph, "cox_model.pkl")
    print("✅ Cox 模型已保存为 cox_model.pkl")
    print("Train censoring ratio:", 1 - df_train["composite_event"].mean())
    print("Valid censoring ratio:", 1 - df_ext["composite_event"].mean())

if __name__ == '__main__':
    run_cox_with_external_validation(
        train_file="data6_副本.csv",
        ext_file="datahx1.csv",
        features=available_features,
        test_size=0.0,
        random_state=42,
        time_point=730,
        threshold=0.637,
        out=1
    )


available_features = [
    'PT', 'child分期_A', 'AFP_greater_400', '失血量', '肿瘤是否巨块型分化_1',
    'AST', 'ALT', 'WBC', 'INR', 'TBIL', '年龄__y', '性别_1', 'ALBI',
    '包膜是否受侵犯_未浸及', '肿瘤直径', 'ALB', '是否合并肝硬化_1', '是否大范围切除_1',
    '是否合并肝炎_1', '肿瘤MVI_M0', 'AFP_less_400'
]