import numpy as np
import torch

def predict_proba(model, X):

    device = next(model.parameters()).device

    model.eval()

    with torch.no_grad():

        X = X.to(device)

        logits = model(X)

        probs_pos = torch.sigmoid(logits)      # P(y=1)
        probs_neg = 1 - probs_pos              # P(y=0)

        probs = torch.stack([probs_neg, probs_pos], dim=1)

    return probs.detach().cpu().numpy()

# ========= #
#  VANILLA  #
# ========= #

def lac_score(model, X, y):

    proba = predict_proba(model, X)

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    y01 = ((y + 1) // 2).astype(int)

    return 1 - proba[np.arange(len(y01)), y01]

def calibration_scores_vanilla(model, X, y, s):
    scores = lac_score(model, X, y)

    scores_global = scores
    scores_female = scores[s == 0]
    scores_male = scores[s == 1]

    return scores_global, scores_female, scores_male

# ============================================ #
#  COUNTERFACTUALLY FAIR CONFORMAL PREDICTION  #
# ============================================ #

def aggregate_scores(score_f, score_cf, agg="max"):
    """
    score_f: numpy array (n,)
    score_cf: numpy array (n,)
    agg: "max", "min", "mean"
    """
    
    if agg == "max":
        return np.maximum(score_f, score_cf)
    
    elif agg == "min":
        return np.minimum(score_f, score_cf)
    
    elif agg == "mean":
        return 0.5 * (score_f + score_cf)
    
    else:
        raise ValueError("agg must be 'max', 'min', or 'mean'")

def calibration_scores_lac_agg(model, X, X_cf, y, s, agg):
    
    scores = lac_score(model, X, y)
    scores_cf = lac_score(model, X_cf, y)

    scores_f = scores[s == 0]
    scores_m = scores[s == 1]

    scores_cf_f = scores_cf[(1 - s) == 0]
    scores_cf_m = scores_cf[(1 - s) == 1]

    scores_global = aggregate_scores(scores, scores_cf, agg)
    scores_female = aggregate_scores(scores_f, scores_cf_m, agg)
    scores_male = aggregate_scores(scores_m, scores_cf_f, agg)

    return scores_global, scores_female, scores_male       
