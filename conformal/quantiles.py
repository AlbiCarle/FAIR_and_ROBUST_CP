import numpy as np

def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(scores, q_level, method="higher")