import re, joblib, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, roc_auc_score, brier_score_loss,confusion_matrix
from lifelines.utils         import concordance_index
from sklearn.utils import resample
# ---------- 1. 工具函数 ----------

def bootstrap_cindex_ci(t, y_true, risk_scores, n_bootstrap=1000, alpha=0.95, seed=42):
    rng = np.random.RandomState(seed)
    cidxs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            c = concordance_index(t[idx], 1-risk_scores[idx], y_true[idx])
            cidxs.append(c)
        except:
            continue
    lower = np.percentile(cidxs, (1-alpha)/2*100)
    upper = np.percentile(cidxs, (1+alpha)/2*100)
    return np.mean(cidxs), lower, upper

def bootstrap_brier_ci(y_true, y_prob, n_bootstrap=1000, alpha=0.95, seed=42):
    rng = np.random.RandomState(seed)
    briers = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            b = brier_score_loss(y_true[idx], y_prob[idx])
            briers.append(b)
        except:
            continue
    lower = np.percentile(briers, (1-alpha)/2*100)
    upper = np.percentile(briers, (1+alpha)/2*100)
    return np.mean(briers), lower, upper


def auc_ci(y_true, y_prob, n_bootstrap=1000, seed=42, alpha=0.95):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:  # 保证采样后阳性阴性都有
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lower = np.percentile(aucs, (1-alpha)/2*100)
    upper = np.percentile(aucs, (1+(alpha))/2*100)
    return np.mean(aucs), lower, upper

def extract_numeric(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        m = re.search(r"[\d\.]+", str(x))
        return float(m.group()) if m else np.nan

def load_and_preprocess(path):
    df = pd.read_csv(path).map(extract_numeric)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    df['composite_event'] = ((df['术后复发（是=1，否=0）'] == 1) | (df['target'] == 1)).astype(int)
    return df

def downsample_positive(df, event_col, target_rate, rnd=42):
    neg, pos = df[df[event_col]==0], df[df[event_col]==1]
    n_pos = min(len(pos), int(len(neg) * target_rate/(1-target_rate)))
    pos_s  = pos.sample(n=n_pos, random_state=rnd)
    return pd.concat([neg, pos_s]).sample(frac=1, random_state=rnd).reset_index(drop=True)

def prepare_xy(df, feats):
    X = df[feats].astype(np.float32).values
    y = df["composite_event"].values.astype(np.int64)
    t = df["时间段"].values.astype(np.float32)
    return X, y, t

# ---------- 2. 模型定义 ----------
class DeepMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),      nn.BatchNorm1d(32),  nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

# ---------- 3. 训练过程 ----------
def train_nn(X, y, val_ratio=0.1, lr=1e-3, epoch=1000, bs=128, seed=42):
    torch.manual_seed(seed)
    model = DeepMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=seed)
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train.reshape(-1, 1)).float()
    X_val,   y_val   = torch.tensor(X_val),   torch.tensor(y_val.reshape(-1, 1)).float()

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    best_loss = float("inf"); patience, wait = 15, 0
    for ep in range(epoch):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = loss_fn(val_out, y_val).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict(best_model)
    return model

def predict_prob(model, X):
    model.eval()
    X = np.asarray(X, dtype=np.float32)              # 保证 float32
    with torch.no_grad():
        t = torch.from_numpy(X).to('cpu')            # 明确在 CPU
        model = model.to('cpu')                      # 模型也在 CPU
        out = model(t).squeeze()
        return torch.sigmoid(out).cpu().numpy()      # CPU 上转 numpy

# ---------- 4. 主流程 ----------
def run_deepsurv(train_path, test_path, features,
                 time_point=730, threshold=0.6, rnd=42, verbose=1):

    df_tr = load_and_preprocess(train_path)
    df_te = load_and_preprocess(test_path)

    # p_train = df_tr["composite_event"].mean()
    # df_te   = downsample_positive(df_te, "composite_event", p_train, rnd)

    X_tr, y_tr, t_tr = prepare_xy(df_tr, features)
    X_te, y_te, t_te = prepare_xy(df_te, features)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    model = train_nn(X_tr, y_tr, seed=rnd)

    prob_tr = predict_prob(model, X_tr)
    prob_te = predict_prob(model, X_te)
    pred_tr = (prob_tr > threshold).astype(int)
    pred_te = (prob_te > threshold).astype(int)

    acc_tr  = accuracy_score(y_tr, pred_tr); acc_te = accuracy_score(y_te, pred_te)
    auc_tr  = roc_auc_score(y_tr, prob_tr);  auc_te = roc_auc_score(y_te, prob_te)
    bri_tr  = brier_score_loss(y_tr, prob_tr); bri_te = brier_score_loss(y_te, prob_te)
    cidx_tr = concordance_index(t_tr,1-prob_tr, y_tr); cidx_te = concordance_index(t_te, 1-prob_te, y_te)
    cm1 = confusion_matrix(y_tr, pred_tr)
    cm = confusion_matrix(y_te, pred_te)
    # if verbose:
    #     print(f"Train  Acc {acc_tr:.3f}  AUC {auc_tr:.3f}  Brier {bri_tr:.3f}  C-index {cidx_tr:.3f}")
    #     print(f"Test   Acc {acc_te:.3f}  AUC {auc_te:.3f}  Brier {bri_te:.3f}  C-index {cidx_te:.3f}")
    #     print(cm)

    # torch.save({"state": model.state_dict(), "scaler": scaler}, "deepsurv_model_cpu.pth")
    pd.DataFrame({"True_Composite_Event": y_te, "Predicted_Event_Probability": prob_te, "Predicted_Class": pred_te}).to_csv("deepsurv_test_results.csv", index=False)
    pd.DataFrame({"True_Composite_Event": y_tr, "Predicted_Event_Probability": prob_tr, "Predicted_Class": pred_tr}).to_csv(
        "deepsurv_train_results.csv", index=False)
    # ---------- AUC 95% CI ----------
    auc_mean_tr, auc_lo_tr, auc_hi_tr = auc_ci(y_tr, prob_tr)
    auc_mean_te, auc_lo_te, auc_hi_te = auc_ci(y_te, prob_te)

    # ---------- Bootstrap CI ----------
    cidx_mean_tr, cidx_lo_tr, cidx_hi_tr = bootstrap_cindex_ci(t_tr, y_tr, prob_tr)
    cidx_mean_te, cidx_lo_te, cidx_hi_te = bootstrap_cindex_ci(t_te, y_te, prob_te)
    brier_mean_tr, brier_lo_tr, brier_hi_tr = bootstrap_brier_ci(y_tr, prob_tr)
    brier_mean_te, brier_lo_te, brier_hi_te = bootstrap_brier_ci(y_te, prob_te)
    auc_mean_tr, auc_lo_tr, auc_hi_tr = auc_ci(y_tr, prob_tr)
    auc_mean_te, auc_lo_te, auc_hi_te = auc_ci(y_te, prob_te)

    if verbose:
        print(f"Train  Acc {acc_tr:.4f}  AUC {auc_tr:.4f} [{auc_lo_tr:.4f}-{auc_hi_tr:.4f}] "
              f"Brier {bri_tr:.4f} [{brier_lo_tr:.4f}-{brier_hi_tr:.4f}] "
              f"C-index {cidx_tr:.4f} [{cidx_lo_tr:.4f}-{cidx_hi_tr:.4f}]")
        print(f"Test   Acc {acc_te:.4f}  AUC {auc_te:.4f} [{auc_lo_te:.4f}-{auc_hi_te:.4f}] "
              f"Brier {bri_te:.4f} [{brier_lo_te:.4f}-{brier_hi_te:.4f}] "
              f"C-index {cidx_te:.4f} [{cidx_lo_te:.4f}-{cidx_hi_te:.4f}]")
        print("train:", cm1)
        print(cm)

    from nn_wrapper import ProbWrapper

    wrapper = ProbWrapper(model, scaler=scaler, features=features)
    joblib.dump(wrapper, 'deepsurv_model.joblib', compress=3)


    return {'train': (acc_tr, auc_tr, bri_tr, cidx_tr),
            'test':  (acc_te, auc_te, bri_te, cidx_te)}

# ---------- 5. 示例调用 ----------
if __name__ == "__main__":
    feats = [
        'PT','child分期_A','AFP_greater_400','失血量','肿瘤是否巨块型分化_1','AST','ALT',
        'WBC','INR','TBIL','年龄__y','性别_1','ALBI','包膜是否受侵犯_未浸及','肿瘤直径',
        'ALB','是否合并肝硬化_1','是否大范围切除_1','是否合并肝炎_1','肿瘤MVI_M0','AFP_less_400'
    ]
    run_deepsurv("data6_副本.csv", "datahx1.csv", feats, time_point=730, threshold=0.35)
