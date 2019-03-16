import numpy as np

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

def construct_forced_samples(N, K, q, T):
    """
    Constructs forced samples for each arm.
    T[i, t] == 1 if arm i is forced sampled for time / sample t,
    otherwise, T[i, t] == 0.

    Arguments:
        N (int): Number of timesteps / samples
        K (int): Number of arms / actions
        q (int): Sampling parameter
        T (np.array): Numpy array of shape (K, N) filled with zeros
    """
    for i in range(K):
        # n = {0, 1, 2, 3, 4, ...}
        n_idxs = np.arange(q)
        # j = {q*i, q*i + 1, q*i + 2, ..., q*i}
        j_idxs = np.arange(q*i, q*(i+1))
        # sampled indices = (2^n - 1)*Kq + j
        idxs = np.array([np.power(2, n_idxs[n])-1 for n in n_idxs])
        idxs = idxs * (K*q) + j_idxs
        # Remove out of bound indices
        idxs = idxs[idxs < N]

        T[i, idxs] = 1

    return T
