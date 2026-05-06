# +
import numpy as np

from conformal.quantiles import *
from conformal.metrics import *
from conformal.scores import *
from conformal.utils import *

def compute_quantiles_and_scaling(model,dataset, X_cal, X_cal_cf, X_test, y_test, s_test,
                                 s_index, alpha, agg):

    # ===== Vanilla Calibration scores =====

    scores_vanilla_g, scores_vanilla_f, scores_vanilla_m = calibration_scores_vanilla(
        model, X_cal,
        dataset.y_cal, dataset.s_cal
    )

    # ===== Calibration scores =====
    scores_g, scores_f, scores_m = calibration_scores_lac_agg(
        model, X_cal, X_cal_cf, dataset.y_cal, dataset.s_cal, agg
    )

    # ===== Quantiles =====
    
    q_vanilla_g = conformal_quantile(scores_vanilla_g, alpha)
    q_vanilla_f = conformal_quantile(scores_vanilla_f, alpha)
    q_vanilla_m = conformal_quantile(scores_vanilla_m, alpha)

    q_g = conformal_quantile(scores_g, alpha)
    q_f = conformal_quantile(scores_f, alpha)
    q_m = conformal_quantile(scores_m, alpha)

    # ===== Base CP sets (for scaling target) =====
    cp_f = cp_sets_lac_agg(model, X_test[s_test == 0], q_f, s_index, agg)
    cp_m = cp_sets_lac_agg(model, X_test[s_test == 1], q_m, s_index, agg)

    E_f = estimate_avg_size(cp_f)
    E_m = estimate_avg_size(cp_m)

    target = max(E_f, E_m)

    # ===== Scaling =====
    k_f = binary_search_k(model, X_test[s_test == 0], q_f, s_index, agg,
                         target, k_min=1.0, k_max=2.0, iters=20)

    k_m = binary_search_k(model, X_test[s_test == 1], q_m, s_index, agg,
                         target, k_min=1.0, k_max=2.0, iters=20)

    # ===== Scaled quantiles =====
    q_f_scaled = k_f * q_f
    q_m_scaled = k_m * q_m
    q_g_scaled = q_g

    # ===== Max =====
    q_max = max(q_f, q_m)

    return {
        "vanilla": (q_vanilla_g, q_vanilla_f, q_vanilla_m),
        "base": (q_g, q_f, q_m, q_max),
        "scaled": (q_g_scaled, q_f_scaled, q_m_scaled)
    }
