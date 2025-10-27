from sklearn.metrics import roc_auc_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

def compute_metrics(y_true, y_pred):


    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    def ci_proportion(successes, total):
        lower, upper = proportion_confint(successes, total, alpha=0.05, method='wilson')
        return (round(lower,4), round(upper,4))

    print(f"✅ Sensitivity: {sensitivity:.4f} 95% CI: {ci_proportion(tp, tp + fn)}")
    print(f"✅ Specificity: {specificity:.4f} 95% CI: {ci_proportion(tn, tn + fp)}")
    print(f"✅ PPV: {ppv:.4f} 95% CI: {ci_proportion(tp, tp + fp)}")
    print(f"✅ NPV: {npv:.4f} 95% CI: {ci_proportion(tn, tn + fn)}")