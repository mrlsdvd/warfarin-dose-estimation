import numpy as np
from sklearn.metrics import accuracy_score

class Evaluator():
    """
    Wrapper class for evaluating a trained model on data with
    different metrics.

    Arguments:
        model (DosageModel): Trained dosage model
    """

    def __init__(self, model):
        self.model = model

    def accuracy(self, predicted, actual):
        """
        Compute accuracy score on given data.

        Arguments:
            actual (np.array): Actual dosages for each example in batch
            predicted (np.array): Predicted dosages for each example in batch

        Returns:
            accuracy (float): Accuracy score
        """
        return accuracy_score(actual, predicted)

    def regret(self, predicted, actual):
        regret = np.zeros((predicted.shape[0]+1, ))
        total_regret = 0
        for i in range(predicted.shape[0]):
            if predicted[i] != actual[i]:
                total_regret += 1
            regret[i+1] = total_regret
        return regret

    def incorrect_decisions(self, predicted, actual):
        incorrect_decisions = np.zeros(predicted.shape)
        num_incorrect_decisions = 0
        for i in range(predicted.shape[0]):
            if predicted[i] != actual[i]:
                num_incorrect_decisions += 1
            incorrect_decisions[i] = num_incorrect_decisions/(i+1)
        return incorrect_decisions
