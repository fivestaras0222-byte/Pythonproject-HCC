import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

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

def load_and_preprocess_data(file_path="datahx1.csv"):
    df = pd.read_csv(file_path)
    df = df.map(extract_numeric)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    df["composite_event"] = ((df["target"] == 1) | (df["术后复发（是=1，否=0）"] == 1)).astype(int)
    return df

def run_st_feature_selection(file_path="data6_副本.csv", test_size=0.21, random_state=42, selection_ratio=0.9):
    df = load_and_preprocess_data(file_path)
    X = df[['是否合并肝硬化_1', 'AST', 'ALT', '是否开腹手术（1=开腹，无=腹腔镜手术）_1',
                                             'ALP', '是否大范围切除_1', 'ALBI', 'WBC', 'PT', 'INR', 'AFP_less_400', 'ALB',
                                             '肿瘤是否巨块型分化_1', 'child分期_A','性别_1','AFP_greater_400','肿瘤直径','是否合并肝炎_1','肿瘤MVI_M0','包膜是否受侵犯_未浸及']]
    # y_true = df["composite_event"]
    y_true = df["术后复发（是=1，否=0）"]

    X_train, _, y_train, _ = train_test_split(
        X, y_true, test_size=test_size, random_state=random_state, stratify=y_true
    )
    nvalue1 = 200
    nvalue2 = 260
    nvalue3 = 190
    rf = RandomForestClassifier(n_estimators=nvalue1, random_state=random_state)
    gbdt = GradientBoostingClassifier(n_estimators=nvalue2, random_state=random_state)
    et = ExtraTreesClassifier(n_estimators=nvalue3, random_state=random_state)

    rf.fit(X_train, y_train)
    gbdt.fit(X_train, y_train)
    et.fit(X_train, y_train)

    avg_importance = (rf.feature_importances_ + gbdt.feature_importances_ + et.feature_importances_) / 3
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": avg_importance
    }).sort_values(by="Importance", ascending=False)

    n_features = len(importance_df)
    n_select = int(n_features * selection_ratio)
    selected_features = importance_df.iloc[:n_select]["Feature"].tolist()
    dropped_features = importance_df.iloc[n_select:]["Feature"].tolist()
    print(selected_features)

    selected_features_df = pd.DataFrame({'Selected_Feature': selected_features})
    selected_features_df.to_csv("selected_features_hx.csv", index=False)

if __name__ == '__main__':
    run_st_feature_selection()
