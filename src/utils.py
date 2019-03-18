import numpy as np
import scipy.stats

def bin_predictions(predictions):
    """
    Bucket the dosage into:
        0 for under 3mg/day,
        1 for 3-7 mg/day, and
        2 for over 7 mg/day

    Arguments:
        predictions (np.array): Array of shape (N, 1) of continuous predictions.
            These predictions are of the square root of weekly warfarin dose.

    Returns:
        binned_predictions (np.array): Array of shape (N, 1) of binned predictions
    """
    # Square all values
    weekly_dosage = np.power(predictions, 2)
    # Get daily dosage
    daily_dosage = weekly_dosage / 7

    daily_dosage[daily_dosage < 3] = 0
    daily_dosage[(daily_dosage >= 3) & (daily_dosage <= 7)] = 1
    daily_dosage[daily_dosage > 7] = 2

    return daily_dosage


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
