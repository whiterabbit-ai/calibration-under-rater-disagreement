import numpy as np

def model_and_human_probs_from_scores(Y, p_scores):
    """
    Generate aligned model and human probability distributions
    for binary classification.

    Inputs:
        Y: shape (n_samples, n_rads) with {0,1}
        p_scores: shape (n_samples,)
    """
    human_prob = Y.mean(axis=1)
    model_prob = p_scores
    return model_prob, human_prob

def dataset_tvd_binary(human_prob, model_prob):
    return np.abs(model_prob - human_prob).mean()
