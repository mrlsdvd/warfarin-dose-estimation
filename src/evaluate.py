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

    def accuracy(self, data, actual):
        """
        Compute accuracy score on given data.

        Arguments:
            data (np.array): Batch of examples to evaluate model on
            actual (np.array): Actual dosages for each example in batch

        Returns:
            accuracy (float): Accuracy score
        """
        predicted = self.model.predict(data)
        return accuracy_score(actual, predicted)
