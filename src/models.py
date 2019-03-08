import numpy as np
import pandas as pd

class DosageBaseline():
    def __init__(self):
        pass

    def predict(self, X):
        """
        Estimate the square root of weekly warfarin dose.

        Arguments:
            X (np.array): Batch of inputs of shape (N, D), where D is the
                number of features (those defined above)

        Returns:
            Y (np.array): Batch of square root weekly dose estimates of
                shape (N, 1)
        """
        N, D = X.shape
        # Add offset bias as 1
        ones = np.ones((N, 1))
        offset_data = np.hstack([ones, X])
        Y = np.dot(offset_data, self.weights)
        return Y

class PDABaseline(DosageBaseline):
    """
    Implements Pharmacogenetic dosing algorithm.
    Input should contain features (in order):
        Age in decades (int in [0,9])
        Height in cm (float)
        Weight in kg (float)
        VKORC1^ A/G (int either 1 or 0)
        VKORC1 A/A (int either 1 or 0)
        VKORC1 genotype unknown (int either 1 or 0)
        CYP2C9 *1/*2 (int either 1 or 0)
        CYP2C9 *1/*3 (int either 1 or 0)
        CYP2C9 *2/*2 (int either 1 or 0)
        CYP2C9 *2/*3 (int either 1 or 0)
        CYP2C9 *3/*3 (int either 1 or 0)
        CYP2C9 genotype unknown (int either 1 or 0)
        Asian race (int either 1 or 0)
        Black or African American (int either 1 or 0)
        Missing or Mixed race (int either 1 or 0)
        Enzyme inducer status (int either 1 or 0)
        Amiodarone status (int either 1 or 0)
    """

    def __init__(self):
        weights = [
            5.6044,     # Offset
            -0.2614,    # Age
            0.0087,     # Height
            0.0128,     # Weight
            -0.8677,    # VKORC1 A/G
            -1.6974,    # VKORC1 A/A
            -0.4854,    # VKORC1 unknown
            -0.5211,    # CYP2C9 *1/*2
            -0.9357,    # CYP2C9 *1/*3
            -1.0616,    # CYP2C9 *2/*2
            -1.9206,    # CYP2C9 *2/*3
            -2.3312,    # CYP2C9 *3/*3
            -0.2188,    # CYP2C9 unknown
            -0.1092,    # Asian
            -0.2760,    # Black or African American
            -0.1032,    # Race missing
            1.1816,     # Enzyme inducer
            -0.5503     # Amiodarone
        ]

        self.weights = np.array(weights)


class CDABaseline(DosageBaseline):
    """
    Implements Clinical dosing algorithm.
    Input should contain features (in order):
        Age in decades (int in [0,9])
        Height in cm (float)
        Weight in kg (float)
        Asian race (int either 1 or 0)
        Black or African American (int either 1 or 0)
        Missing or Mixed race (int either 1 or 0)
        Enzyme inducer status (int either 1 or 0)
        Amiodarone status (int either 1 or 0)
    """

    def __init__(self):
        weights = [
            4.0376,     # Offset
            -0.2546,    # Age
            0.0118,     # Height
            0.0134,     # Weights
            -0.6752,    # Asian
            0.4060,     # Black or African American
            0.0443,     # Race missing or mixed
            1.2799,     # Enzyme inducer
            -0.5695     # Amiodarone
        ]

        self.weights = np.array(weights)

class FDBaseline(DosageBaseline):
    """
    Implements fixed dosage baseline.
    Regardless of input, a fixed dosage is returned
    for each instance in the batch.
    """

    def __init__(self, dosage):
        self.dosage = dosage

    def predict(self, X):
        N, D = X.shape
        Y = np.ones((N, 1))
        return Y
