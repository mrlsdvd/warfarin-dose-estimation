import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

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


def plot_regret(linear_UCB_regret, num_examples, plot_title, plot_x_axis_label, 
                plot_y_axis_label, main_line_label, confidence_interval_label):
    # Plot Linear UCB regret
    mean_linear_UCB_regret = np.zeros((num_examples+1,))
    lower_bound_linear_UCB_regret = np.zeros((num_examples+1,))
    upper_bound_linear_UCB_regret = np.zeros((num_examples+1,))

    for i in range(linear_UCB_regret.shape[1]):
        mean, lower_bound, upper_bound = mean_confidence_interval(linear_UCB_regret[:,i], confidence=0.95)
        mean_linear_UCB_regret[i] = mean
        lower_bound_linear_UCB_regret[i] = lower_bound
        upper_bound_linear_UCB_regret[i] = upper_bound

    plt.clf()
    plt.plot(range(mean_linear_UCB_regret.shape[0]), mean_linear_UCB_regret, lw=1, color='#539caf', 
             alpha=1, label=main_line_label)
    plt.fill_between(range(lower_bound_linear_UCB_regret.shape[0]), lower_bound_linear_UCB_regret, 
                     upper_bound_linear_UCB_regret, color='#539caf', alpha=0.4, label=confidence_interval_label)
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_label)
    plt.ylabel(plot_y_axis_label)
    plt.legend()
    plt.show()
    
    
def plot_fraction_incorrect(linear_UCB_incorrect, num_examples, plot_title, plot_x_axis_label, 
                plot_y_axis_label, main_line_label, confidence_interval_label):
    x_axis = []
    for i in range(linear_UCB_incorrect.shape[1]-1):
        x_axis.append(i*100)
    x_axis.append(num_examples)


    mean_linear_UCB_incorrect = np.zeros((int(np.ceil(num_examples/100.0)+1,)))
    lower_bound_linear_UCB_incorrect = np.zeros((int(np.ceil(num_examples/100.0)+1,)))
    upper_bound_linear_UCB_incorrect = np.zeros((int(np.ceil(num_examples/100.0)+1,)))

    for i in range(linear_UCB_incorrect.shape[1]):
        mean, lower_bound, upper_bound = mean_confidence_interval(linear_UCB_incorrect[:,i], confidence=0.95)
        mean_linear_UCB_incorrect[i] = mean
        lower_bound_linear_UCB_incorrect[i] = lower_bound
        upper_bound_linear_UCB_incorrect[i] = upper_bound

    plt.clf()
    plt.plot(x_axis, mean_linear_UCB_incorrect, lw=1, color='#539caf', alpha=1, label=main_line_label)
    plt.fill_between(x_axis, lower_bound_linear_UCB_incorrect, upper_bound_linear_UCB_incorrect, 
                     color='#539caf', alpha=0.4, label=confidence_interval_label)
    plt.title(plot_title)
    plt.xlabel(plot_x_axis_label)
    plt.ylabel(plot_y_axis_label)
    plt.legend()
    plt.show()

    