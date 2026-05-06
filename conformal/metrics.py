# +
import numpy as np

from .scores import *
from .quantiles import *
from .sets import *

def percp_stats(y_test, conformal_sets, nsplits=50, verbose=True):

    if torch.is_tensor(y_test):
        y_test = y_test.detach().cpu().numpy()

    # conversion {-1,+1} → {0,1}
    y_test = ((y_test + 1) // 2).astype(int)

    n = len(y_test)

    mask = np.array([y_test[i] in conformal_sets[i] for i in range(n)])
    coverage = mask.mean()
    coverage_std = mask.std()

    coverage_ci = 1.96 * coverage_std / np.sqrt(n) #nsplits 

    sizes = np.array([len(s) for s in conformal_sets])
    avg_size = sizes.mean()
    std_size = sizes.std()
    size_ci = 1.96 * std_size / np.sqrt(n) # 1.96

    if verbose:
        print(
            f"Coverage = {coverage*100:.2f}% "
            f"± {coverage_ci*100:.2f}% (95% CI)"
        )
        print(
            f"Set Size = {avg_size:.2f} "
            f"± {size_ci:.2f} (95% CI)"
        )

    return coverage, coverage_ci, avg_size, size_ci, sizes


# -

def compute_metrics(model, dataset, X_adv, X_cf_adv, q_g, q_f, q_m, agg="max"):

    # Factual
    cp_global = cp_sets_lac_agg(model, X_adv, q_g, dataset.s_index, agg)
    cp_female = cp_sets_lac_agg(model, X_adv[dataset.s_test == 0], q_f, dataset.s_index, agg)
    cp_male   = cp_sets_lac_agg(model, X_adv[dataset.s_test == 1], q_m, dataset.s_index, agg)

    # Counterfactual
    cp_cf_global = cp_sets_lac_agg(model, X_cf_adv, q_g, dataset.s_index, agg)
    cp_cf_female = cp_sets_lac_agg(model, X_cf_adv[1 - dataset.s_test == 0], q_m, dataset.s_index, agg)
    cp_cf_male   = cp_sets_lac_agg(model, X_cf_adv[1 - dataset.s_test == 1], q_f, dataset.s_index, agg)

    # Metrics factual
    cov_g, cov_g_ci, size_g, size_g_ci, _ = percp_stats(dataset.y_test, cp_global, verbose=False)
    cov_f, cov_f_ci, size_f, size_f_ci, _ = percp_stats(dataset.y_test[dataset.s_test == 0], cp_female, verbose=False)
    cov_m, cov_m_ci, size_m, size_m_ci, _ = percp_stats(dataset.y_test[dataset.s_test == 1], cp_male, verbose=False)

    # Metrics counterfactual
    cov_cf_g, cov_cf_g_ci, size_cf_g, size_cf_g_ci, _ = percp_stats(dataset.y_test, cp_cf_global, verbose=False)
    cov_cf_f, cov_cf_f_ci, size_cf_f, size_cf_f_ci, _ = percp_stats(dataset.y_test[1 - dataset.s_test == 0], cp_cf_female, verbose=False)
    cov_cf_m, cov_cf_m_ci, size_cf_m, size_cf_m_ci, _ = percp_stats(dataset.y_test[1 - dataset.s_test == 1], cp_cf_male, verbose=False)

    return {
        "cov": (cov_g, cov_f, cov_m),
        "cov_cf": (cov_cf_g, cov_cf_f, cov_cf_m),
        "size": (size_g, size_f, size_m),
        "size_cf": (size_cf_g, size_cf_f, size_cf_m),
        "ci": {
            "cov": (cov_g_ci, cov_f_ci, cov_m_ci),
            "cov_cf": (cov_cf_g_ci, cov_cf_f_ci, cov_cf_m_ci),
            "size": (size_g_ci, size_f_ci, size_m_ci),
            "size_cf": (size_cf_g_ci, size_cf_f_ci, size_cf_m_ci),
        }
    }

# +
def compute_metrics_vanilla(model, dataset, X_adv, X_cf_adv, q_g, q_f, q_m):

    # Factual
    cp_global = cp_sets_lac_vanilla(model, X_adv, q_g)
    cp_female = cp_sets_lac_vanilla(model, X_adv[dataset.s_test == 0], q_f)
    cp_male   = cp_sets_lac_vanilla(model, X_adv[dataset.s_test == 1], q_m)

    # Counterfactual
    cp_cf_global = cp_sets_lac_vanilla(model, X_cf_adv, q_g)
    cp_cf_female = cp_sets_lac_vanilla(model, X_cf_adv[1 - dataset.s_test == 0], q_m)
    cp_cf_male   = cp_sets_lac_vanilla(model, X_cf_adv[1 - dataset.s_test == 1], q_f)

    # Metrics factual
    cov_g, cov_g_ci, size_g, size_g_ci, _ = percp_stats(dataset.y_test, cp_global, verbose=False)
    cov_f, cov_f_ci, size_f, size_f_ci, _ = percp_stats(dataset.y_test[dataset.s_test == 0], cp_female, verbose=False)
    cov_m, cov_m_ci, size_m, size_m_ci, _ = percp_stats(dataset.y_test[dataset.s_test == 1], cp_male, verbose=False)

    # Metrics counterfactual
    cov_cf_g, cov_cf_g_ci, size_cf_g, size_cf_g_ci, _ = percp_stats(dataset.y_test, cp_cf_global, verbose=False)
    cov_cf_f, cov_cf_f_ci, size_cf_f, size_cf_f_ci, _ = percp_stats(dataset.y_test[1 - dataset.s_test == 0], cp_cf_female, verbose=False)
    cov_cf_m, cov_cf_m_ci, size_cf_m, size_cf_m_ci, _ = percp_stats(dataset.y_test[1 - dataset.s_test == 1], cp_cf_male, verbose=False)

    return {
        "cov": (cov_g, cov_f, cov_m),
        "cov_cf": (cov_cf_g, cov_cf_f, cov_cf_m),
        "size": (size_g, size_f, size_m),
        "size_cf": (size_cf_g, size_cf_f, size_cf_m),
        "ci": {
            "cov": (cov_g_ci, cov_f_ci, cov_m_ci),
            "cov_cf": (cov_cf_g_ci, cov_cf_f_ci, cov_cf_m_ci),
            "size": (size_g_ci, size_f_ci, size_m_ci),
            "size_cf": (size_cf_g_ci, size_cf_f_ci, size_cf_m_ci),
        }
    }

def print_table(res, title):

    print(f"\n===== {title} =====")

    for method, r in res.items():

        cov_g, cov_f, cov_m = r["cov"]
        cov_cf_g, cov_cf_f, cov_cf_m = r["cov_cf"]

        size_g, size_f, size_m = r["size"]
        size_cf_g, size_cf_f, size_cf_m = r["size_cf"]

        ci = r["ci"]

        cov_g_ci, cov_f_ci, cov_m_ci = ci["cov"]
        cov_cf_g_ci, cov_cf_f_ci, cov_cf_m_ci = ci["cov_cf"]

        size_g_ci, size_f_ci, size_m_ci = ci["size"]
        size_cf_g_ci, size_cf_f_ci, size_cf_m_ci = ci["size_cf"]

        print(f"\n--- {method} ---")

        print(
            "Coverage     : "
            f"G={cov_g:.2f}±{cov_g_ci:.2f}, "
            f"F={cov_f:.2f}±{cov_f_ci:.2f}, "
            f"M={cov_m:.2f}±{cov_m_ci:.2f}"
        )

        print(
            "Coverage CF  : "
            f"G={cov_cf_g:.2f}±{cov_cf_g_ci:.2f}, "
            f"F={cov_cf_f:.2f}±{cov_cf_f_ci:.2f}, "
            f"M={cov_cf_m:.2f}±{cov_cf_m_ci:.2f}"
        )

        print(
            "Size         : "
            f"G={size_g:.2f}±{size_g_ci:.2f}, "
            f"F={size_f:.2f}±{size_f_ci:.2f}, "
            f"M={size_m:.2f}±{size_m_ci:.2f}"
        )

        print(
            "Size CF      : "
            f"G={size_cf_g:.2f}±{size_cf_g_ci:.2f}, "
            f"F={size_cf_f:.2f}±{size_cf_f_ci:.2f}, "
            f"M={size_cf_m:.2f}±{size_cf_m_ci:.2f}"
        )
