import re
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from lifelines.utils import concordance_index

def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, alpha=0.95, random_state=42):
    """
    通用 bootstrap CI
    metric_fn: metric_fn(y_true_subset, y_pred_subset) -> float
    """
    rng = np.random.RandomState(random_state)
    scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(scores, (1 + alpha) / 2 * 100)
    return np.mean(scores), lower, upper

def extract_numeric(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        m = re.search(r"[\d\.]+", str(x))
        return float(m.group()) if m else np.nan


def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.map(extract_numeric)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    df['composite_event'] = ((df['术后复发（是=1，否=0）'] == 1) | (df['target'] == 1)).astype(int)
    return df


def downsample_positive(df, event_col, target_rate, random_state=42):
    neg = df[df[event_col] == 0]
    pos = df[df[event_col] == 1]
    n_pos = min(len(pos), int(len(neg) * target_rate / (1 - target_rate)))
    pos_s = pos.sample(n=n_pos, random_state=random_state)
    return pd.concat([neg, pos_s]).sample(frac=1, random_state=random_state).reset_index(drop=True)


def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, alpha=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    aucs = []
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
    return np.mean(aucs), lower, upper


def run_logistic_with_imported_features(
    data_file="data6_副本.csv",
    data_new_file="datahx1.csv",
    features=None,
    test_size=0.0,
    random_state=42,
    time_point=730,
    threshold=0.5,
    out=1
):
    # load
    df_tr = load_and_preprocess(data_file)
    df_va = load_and_preprocess(data_new_file)

    # #downsample validation to match train event rate
    # p_tr = df_tr["composite_event"].mean()
    # df_va = downsample_positive(df_va, "composite_event", p_tr, random_state)

    # create binary label: event within time_point
    df_tr["y_bin"] = ((df_tr["composite_event"] == 1) & (df_tr["时间段"] <= time_point)).astype(int)
    df_va["y_bin"] = ((df_va["composite_event"] == 1) & (df_va["时间段"] <= time_point)).astype(int)

    if features is None:
        features = [c for c in df_tr.columns if c not in ["时间段", "composite_event", "y_bin"]]

    X_tr = df_tr[features].values
    y_tr = df_tr["y_bin"].values
    t_tr = df_tr["时间段"].values

    X_va = df_va[features].values
    y_va = df_va["y_bin"].values
    t_va = df_va["时间段"].values

    # optional train/test split
    if test_size > 0:
        X_tr, X_test, y_tr, y_test, t_tr, _ = train_test_split(
            X_tr, y_tr, t_tr,
            test_size=test_size,
            random_state=random_state,
            stratify=y_tr
        )

    # scale & fit
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    clf = LogisticRegression(
        penalty='l1', C=10000000,
        solver="liblinear",
        random_state=random_state, max_iter=500000000
    )
    clf.fit(X_tr_s, y_tr)
    # evaluation
    def eval_set(X, y, t, tag):
        proba = clf.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        acc = accuracy_score(y, pred)
        auc_mean, auc_lo, auc_hi = bootstrap_metric_ci(y, proba, lambda yt, yp: roc_auc_score(yt, yp))
        bri_mean, bri_lo, bri_hi = bootstrap_metric_ci(y, proba, lambda yt, yp: brier_score_loss(yt, yp))
        cidx_mean, cidx_lo, cidx_hi = bootstrap_metric_ci(t, -proba, lambda tt, pp: concordance_index(tt, pp, y))

        cm = confusion_matrix(y, pred)

        if out:
            print(f"--- {tag} ---")
            print(f"Accuracy: {acc:.4f}")
            print(f"AUC: {auc_mean:.4f} (95% CI {auc_lo:.4f}–{auc_hi:.4f})")
            print(f"Brier score: {bri_mean:.4f} (95% CI {bri_lo:.4f}–{bri_hi:.4f})")
            print(f"Concordance index: {cidx_mean:.4f} (95% CI {cidx_lo:.4f}–{cidx_hi:.4f})")
            print("Confusion matrix:")
            print(cm)
        return proba

    proba_tr = eval_set(X_tr_s, y_tr, t_tr, "Training")
    proba_va = eval_set(X_va_s, y_va, t_va, "Validation")
    pred_tr = (proba_tr >= threshold).astype(int)
    pred_va = (proba_va >= threshold).astype(int)
    # save
    pd.DataFrame({"True_Composite_Event": y_tr, "Predicted_Event_Probability": proba_tr,"Predicted_Class": pred_tr}).to_csv(
        "logistic_train_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"True_Composite_Event": y_va, "Predicted_Event_Probability": proba_va,"Predicted_Class": pred_va}).to_csv(
        "logistic_valid_results.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump((clf, scaler), "logistic_model.pkl")

    return clf, scaler


if __name__ == "__main__":
    available_features = [
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

    run_logistic_with_imported_features(
        data_file="data6_副本.csv",
        data_new_file="datahx1.csv",
        features=available_features,
        test_size=0.0,
        random_state=42,
        time_point=730,
        threshold=0.4,
        out=1
    )
