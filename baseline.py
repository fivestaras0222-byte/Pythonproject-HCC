import pandas as pd
from lifelines import CoxPHFitter

def preprocess_vars(data, normal_ranges=None):
    if normal_ranges:
        for var, (low, high) in normal_ranges.items():
            new_col = f"{var}_group"
            data[new_col] = ((data[var] < low) | (data[var] > high)).astype(int)
    return data
def run_univariate_cox(data, time_col, event_col, covariates):
    results = []
    for covariate in covariates:
        df = data[[time_col, event_col, covariate]].dropna()
        cph = CoxPHFitter()
        try:
            cph.fit(df, duration_col=time_col, event_col=event_col)
            summary = cph.summary
            hr = summary.loc[covariate, 'exp(coef)']
            ci_lower = summary.loc[covariate, 'exp(coef) lower 95%']
            ci_upper = summary.loc[covariate, 'exp(coef) upper 95%']
            p = summary.loc[covariate, 'p']
            results.append({
                'Variable': covariate,
                'HR': hr,
                '95% CI': f"{ci_lower:.3f} - {ci_upper:.3f}",
                'p-value': p
            })
        except Exception as e:
            print(f"Univariate Cox failed for {covariate}: {e}")
    return pd.DataFrame(results)


def run_multivariate_cox(data, time_col, event_col, univariate_results):
    significant_vars = univariate_results[univariate_results['p-value'] < 0.05]['Variable'].tolist()

    df = data[[time_col, event_col] + significant_vars].dropna()
    cph = CoxPHFitter()
    cph.fit(df, duration_col=time_col, event_col=event_col)
    summary = cph.summary

    results = []
    for covariate in significant_vars:
        hr = summary.loc[covariate, 'exp(coef)']
        ci_lower = summary.loc[covariate, 'exp(coef) lower 95%']
        ci_upper = summary.loc[covariate, 'exp(coef) upper 95%']
        p = summary.loc[covariate, 'p']
        results.append({
            'Variable': covariate,
            'HR': hr,
            '95% CI': f"{ci_lower:.3f} - {ci_upper:.3f}",
            'p-value': p
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    data = pd.read_csv('data6_副本改.csv')

    time_col = '时间段'
    event_col = '术后复发（是=1，否=0）'

    normal_ranges = {
        "ALT": (0, 40),
        "AST": (0, 35),
        "TBIL": (0, 20),
        "R-GGT": (7, 35),
        "ALP": (40, 150),
        "PLT": (100, 300),
        "失血量": (0, 800),
        "ALB": (35, 55),
        "INR": (0.8, 1.2),
        "WBC": (4.0, 10.0)
    }
    data = preprocess_vars(data, normal_ranges=normal_ranges)

    covariates = [
        "性别_1", "年龄__y",
        "ALT_group", "AST_group", "TBIL_group",
        "R-GGT_group", "ALP_group", "PLT_group",
        "ALB_group", "PT", "INR_group", "WBC_group",
        "AFP_less_400", "AFP_greater_400", "ALBI",
        "失血量_group",
        "肿瘤直径", "包膜是否受侵犯_未浸及", "肿瘤是否巨块型分化_1", "肿瘤MVI_M0",
        "是否合并肝炎_1", "是否合并肝硬化_1", "child分期_A"
    ]

    univariate_results = run_univariate_cox(data, time_col, event_col, covariates)
    univariate_results.to_excel('univariate_cox_results_c.xlsx', index=False)

    multivariate_results = run_multivariate_cox(data, time_col, event_col, univariate_results)
    multivariate_results.to_excel('multivariate_cox_results_c.xlsx', index=False)