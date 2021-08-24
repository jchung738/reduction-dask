import pandas as pd
import numpy as np


def spearman_rank(y_true, y_pred, era):
    """
    Inputs: y_true (Pandas Series) True targets
            y_pred (array-like)  Predicted targets
            era (Pandas Series) Eras containing
    Output: Spearman Rank Correlation

    Takes in y and y_hat, ranks said values within each corresponding era, then finds the rank correlation between each era.
    """
    # Use index of y_true to form y_pred as DataFrame
    y_pred = pd.DataFrame(y_pred, index=Y_True.index)
    # Join y_pred with the era Series using indices
    y_pred = y_pred.join(era).dropna()
    # Percentile rank each era
    ranked_pred = y_pred.groupby('era').apply(lambda x: x.rank(pct=True, method="first")).values[:, 0]
    # Find correlation between y_true and the ranked predictions
    corr = np.corrcoef(np.ravel(y_true), ranked_pred)[0, 1]
    return corr


def quartic_error(y_true, y_pred):
    """
    Inputs: Y_True (array-like) True targets
            Y_Pred (array-like)  Predicted targets
    Output: Quartic Error

    This function finds the mean error to the fourth power to increase the influence of extreme errors, which is important for financial markets.
    """
    return np.mean((y_true - y_pred) ** 4)
