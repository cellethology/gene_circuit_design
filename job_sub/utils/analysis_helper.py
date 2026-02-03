import numpy as np
from scipy import stats


def mannwhitney_p(
    values, baseline_values, alternative="two-sided", return_effect=False
):
    if len(values) == 0 or len(baseline_values) == 0:
        return (np.nan, np.nan) if return_effect else np.nan
    try:
        res = stats.mannwhitneyu(values, baseline_values, alternative=alternative)
        p_value = float(res.pvalue)
    except ValueError:
        return (np.nan, np.nan) if return_effect else np.nan

    if not return_effect:
        return p_value

    # Probability of superiority (ties count as 0.5).
    n1 = len(values)
    n2 = len(baseline_values)
    combined = np.concatenate([values, baseline_values])
    ranks = stats.rankdata(combined)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    ps = U1 / (n1 * n2)

    return p_value, float(ps)


def fold_change(strategy_median, baseline_median):
    return strategy_median / baseline_median
