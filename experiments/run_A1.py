# +
from tqdm import tqdm

from attacks.pgd_cf import generate_attacked_data
from conformal.quantiles import *
from conformal.metrics import *
from conformal.scores import *
from conformal.utils import *

from datasets.loader import flip_sensitive

def run_counterfactual_experiment_A1(model, dataset, eps_values, lambd, alpha, agg="max"):

    X_cal_cf  = flip_sensitive(dataset.X_cal, dataset.s_index)
    X_test_cf = flip_sensitive(dataset.X_test, dataset.s_index)

    nat_q = compute_quantiles_and_scaling(
        model, dataset,
        dataset.X_cal, X_cal_cf,
        dataset.X_test,
        dataset.y_test,
        dataset.s_test,
        dataset.s_index,
        alpha,
        agg
    )
    
    results = []

    for eps in tqdm(eps_values):

        X_cal_adv, X_cal_cf_adv = generate_attacked_data(
            model, dataset.X_cal, dataset.y_cal,
            dataset.s_cal, dataset.s_index, eps, lambd
        )

        X_test_adv, X_test_cf_adv = generate_attacked_data(
            model, dataset.X_test, dataset.y_test,
            dataset.s_test, dataset.s_index, eps, lambd
        )

        percp_q = compute_quantiles_and_scaling(
            model, dataset,
            X_cal_adv, X_cal_cf_adv,
            X_test_adv,
            dataset.y_test,
            dataset.s_test,
            dataset.s_index,
            alpha,
            agg
        )
        
        # ===== NATURAL =====
        metrics_nat_C = compute_metrics_vanilla(
            model, dataset,
            X_test_adv, X_test_cf_adv,
            nat_q["vanilla"][0],
            nat_q["vanilla"][1],
            nat_q["vanilla"][2]
        )

        # ===== PERCP =====
        metrics_percp_C = compute_metrics_vanilla(
            model, dataset,
            X_test_adv, X_test_cf_adv,
            percp_q["vanilla"][0],
            percp_q["vanilla"][1],
            percp_q["vanilla"][2]
        )

        # ===== NATURAL =====
        metrics_nat_C_EQ = compute_metrics(
            model, dataset,
            X_test_adv, X_test_cf_adv,
            nat_q["base"][0],
            nat_q["base"][1],
            nat_q["base"][2],
            agg
        )

        # ===== PERCP =====
        metrics_percp_C_EQ = compute_metrics(
            model, dataset,
            X_test_adv, X_test_cf_adv,
            percp_q["base"][0],
            percp_q["base"][1],
            percp_q["base"][2],
            agg
        )
        
        # ===== NATURAL =====
        metrics_nat_C_EQ_k_a = compute_metrics(
            model, dataset,
            X_test_adv, X_test_cf_adv,
            nat_q["scaled"][0],
            nat_q["scaled"][1],
            nat_q["scaled"][2],
            agg
        )

        # ===== PERCP =====
        metrics_percp_C_EQ_k_a = compute_metrics(
            model, dataset,
            X_test_adv, X_test_cf_adv,
            percp_q["scaled"][0],
            percp_q["scaled"][1],
            percp_q["scaled"][2],
            agg
        )
        
        # ===== NATURAL =====
        metrics_nat_C_EQ_max = compute_metrics(
            model, dataset,
            X_test_adv, X_test_cf_adv,
            nat_q["base"][3],
            nat_q["base"][3],
            nat_q["base"][3],
            agg
        )

        # ===== PERCP =====
        metrics_percp_C_EQ_max = compute_metrics(
            model, dataset,
            X_test_adv, X_test_cf_adv,
            percp_q["base"][3],
            percp_q["base"][3],
            percp_q["base"][3],
            agg
        )

        results.append({
            "eps": eps,
            "natural_vanilla": metrics_nat_C,
            "percp_vanilla": metrics_percp_C,
            "natural_EQ": metrics_nat_C_EQ,
            "percp_EQ": metrics_percp_C_EQ,
            "natural_EQ_ka": metrics_nat_C_EQ_k_a,
            "percp_EQ_ka": metrics_percp_C_EQ_k_a,
            "natural_EQ_max": metrics_nat_C_EQ_max,
            "percp_EQ_max": metrics_percp_C_EQ_max
        })

    return results
