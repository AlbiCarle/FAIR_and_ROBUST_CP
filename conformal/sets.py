# +
import numpy as np

from .scores import *
from .quantiles import *

def cp_sets_lac_agg(model, X, qhat, sex_index, agg="max"):
    
    if isinstance(X, np.ndarray):
        X_cf = X.copy()
        X_cf[:, sex_index] = 1 - X_cf[:, sex_index]
    elif torch.is_tensor(X):
        X_cf = X.clone()
        X_cf[:, sex_index] = 1 - X_cf[:, sex_index]
    else:
        raise TypeError("X deve essere numpy array o torch tensor")
    
    probs_f  = predict_proba(model, X)
    probs_cf = predict_proba(model, X_cf)
    
    sets = []
    
    for p_f, p_cf in zip(probs_f, probs_cf):
        labels = []
        for y in [0, 1]:
            s_f  = 1.0 - p_f[y]
            s_cf = 1.0 - p_cf[y]
            
            if agg == "max":
                s_agg = max(s_f, s_cf)
            elif agg == "min":
                s_agg = min(s_f, s_cf)
            elif agg == "mean":
                s_agg = 0.5 * (s_f + s_cf)
            else:
                raise ValueError("agg must be 'max', 'min', or 'mean'")
            
            if s_agg <= qhat:
                labels.append(y)
        
        sets.append(set(labels))
    
    return sets


# -

def cp_sets_lac_vanilla(model, X, qhat):
    
    probs = predict_proba(model, X)
    
    sets = []
    
    for p in probs:
        labels = []
        for y in [0, 1]:
            s = 1.0 - p[y]
            
            if s <= qhat:
                labels.append(y)
        
        sets.append(set(labels))
    
    return sets

def cp_sets_lac(model, X, qhat):
    probs = predict_proba(model, X)
    sets = []
    for p in probs:
        labels = np.where(1.0 - p <= qhat)[0]
        sets.append(set(labels))
    return sets

def compare_sets(sets_factual, sets_cf):
    
    n = len(sets_factual)
    identical = 0
    diffs = []
    
    for i, (C_f, C_cf) in enumerate(zip(sets_factual, sets_cf)):
        if C_f == C_cf:
            identical += 1
        else:
            diffs.append(i)
    
    perc_identical = 100 * identical / n
    return perc_identical, diffs


# +
# ============================ #
# EQUALIZED AVERAGE SET SIZE  #
# ============================ #

def estimate_avg_size(cp_sets):
    return np.mean([len(s) for s in cp_sets])

def compute_Ek(model, X, q, s_index, agg):
    cp = cp_sets_lac_agg(model, X, q, s_index, agg)
    return estimate_avg_size(cp)

def find_k_for_target(model, X, q, s_index, agg, target, k_grid=np.linspace(1.0, 2.0, 30)):
    best_k = None
    best_err = float("inf")

    for k in k_grid:
        cp = cp_sets_lac_agg(model, X, k*q, s_index, agg)
        E = estimate_avg_size(cp)

        err = abs(E - target)

        if err < best_err:
            best_err = err
            best_k = k

    return best_k

def binary_search_k(model, X, q, s_index, agg, target, k_min=1.0, k_max=5.0, iters=15):
    for _ in range(iters):
        k_mid = (k_min + k_max) / 2

        cp = cp_sets_lac_agg(model, X, k_mid*q, s_index, agg)
        E = estimate_avg_size(cp)

        if E < target:
            k_min = k_mid
        else:
            k_max = k_mid

    return (k_min + k_max) / 2
# -




