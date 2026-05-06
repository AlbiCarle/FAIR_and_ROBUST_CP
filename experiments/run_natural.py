from datasets.loader import flip_sensitive
from conformal.quantiles import *
from conformal.metrics import *
from conformal.scores import *
from conformal.sets import *
from conformal.utils import *

def run_natural_experiment(model, dataset, X_cal, X_test, alpha, agg="max"):

    X_cal_cf  = flip_sensitive(X_cal, dataset.s_index)
    X_test_cf = flip_sensitive(X_test, dataset.s_index)

    # ===== CALIBRATION =====
    scores_g, scores_f, scores_m = calibration_scores_lac_agg(
        model, X_cal, X_cal_cf,
        dataset.y_cal, dataset.s_cal, agg
    )

    scores_vanilla_g, scores_vanilla_f, scores_vanilla_m = calibration_scores_vanilla(
        model, X_cal,
        dataset.y_cal, dataset.s_cal
    )

    # ===== QUANTILES =====
    q_vanilla_g = conformal_quantile(scores_vanilla_g, alpha)
    q_vanilla_f = conformal_quantile(scores_vanilla_f, alpha)
    q_vanilla_m = conformal_quantile(scores_vanilla_m, alpha)

    q_g = conformal_quantile(scores_g, alpha)
    q_f = conformal_quantile(scores_f, alpha)
    q_m = conformal_quantile(scores_m, alpha)

    # ===== SCALING =====
    cp_female = cp_sets_lac_agg(
        model, X_test[dataset.s_test == 0], q_f, dataset.s_index, agg
    )
    cp_male = cp_sets_lac_agg(
        model, X_test[dataset.s_test == 1], q_m, dataset.s_index, agg
    )

    E_f = estimate_avg_size(cp_female)
    E_m = estimate_avg_size(cp_male)

    target = max(E_f, E_m)

    k_f = binary_search_k(
        model, X_test[dataset.s_test == 0],
        q_f, dataset.s_index, agg, target
    )

    k_m = binary_search_k(
        model, X_test[dataset.s_test == 1],
        q_m, dataset.s_index, agg, target
    )

    q_f_scaled = k_f * q_f
    q_m_scaled = k_m * q_m
    q_g_scaled = q_g

    q_max = max(q_f, q_m)

    # ===== METRICS =====
    res_C = compute_metrics_vanilla(
        model, dataset,
        X_test, X_test_cf,
        q_vanilla_g, q_vanilla_f, q_vanilla_m
    )

    res_C_EQ = compute_metrics(
        model, dataset,
        X_test, X_test_cf,
        q_g, q_f, q_m, agg
    )

    res_C_EQ_k_a = compute_metrics(
        model, dataset,
        X_test, X_test_cf,
        q_g_scaled, q_f_scaled, q_m_scaled, agg
    )

    res_C_EQ_max = compute_metrics(
        model, dataset,
        X_test, X_test_cf,
        q_max, q_max, q_max, agg
    )

    return {
        "vanilla": res_C,
        "EQ": res_C_EQ,
        "EQ_ka": res_C_EQ_k_a,
        "EQ_max": res_C_EQ_max
    }
