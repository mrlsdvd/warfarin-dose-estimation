import numpy as np
import pandas as pd
from utils import bin_predictions

class DosageModel():
    """
    Abstract class for dosage models.
    """

    def __init__(self):
        pass

    def get_features(self, data):
        """
        Extracts necessary feature columns for model from data matrix.

        Arguments:
            data (pd.DataFrame): Data matrix

        Returns:
            features (pd.DataFrame): Feature matrix
        """
        pass

    def get_targets(self, data):
        """
        Extracts targets (labels) from data matrix.

        Arguments:
            data (pd.DataFrame): Data matrix

        Returns:
            targets (pd.DataFrame): Target matrix
        """
        pass

    def predict(self):
        pass

class DosageBaseline(DosageModel):
    def __init__(self):
        pass

    def predict(self, X, bin=True):
        """
        Estimate the square root of weekly warfarin dose.

        Arguments:
            X (np.array): Batch of inputs of shape (N, D), where D is the
                number of features (those defined above)
            bin (bool): Whether to bin the predictions into dosage categories

        Returns:
            Y (np.array): Batch of square root weekly dose estimates of
                shape (N, 1)
        """
        N, D = X.shape
        # Add offset bias as 1
        ones = np.ones((N, 1))
        offset_data = np.hstack([ones, X])
        Y = np.dot(offset_data, self.weights)
        if bin:
            Y = bin_predictions(Y)
        return Y

class PDABaseline(DosageBaseline):
    """
    Implements Pharmacogenetic dosing algorithm.
    Input should contain features (in order):
        Age in decades (int in [0,9])
        Height in cm (float)
        Weight in kg (float)
        VKORC1 A/G (int either 1 or 0)
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
        col_names = [
            "Age",
            "Height_(cm)",
            "Weight_(kg)",
            "VKORC1_A/G",
            "VKORC1_A/A",
            "VKORC1_nan",
            "Cyp2C9_*1/*2",
            "Cyp2C9_*1/*3",
            "Cyp2C9_*2/*2",
            "Cyp2C9_*2/*3",
            "Cyp2C9_*3/*3",
            "Cyp2C9_nan",
            "Race_Asian",
            "Race_Black_or_African_American",
            "Race_nan",
            "enzyme_inducer_status",
            "amiodarone_status"
        ]
        self.col_names = col_names

        self.target_name = 'Therapeutic_Dose_of_Warfarin'

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

    def get_features(self, data):
        return data[self.col_names]

    def get_targets(self, data):
        return data[[self.target_name]]


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
        col_names = [
            "Age",
            "Height_(cm)",
            "Weight_(kg)",
            "Race_Asian",
            "Race_Black_or_African_American",
            "Race_nan",
            "enzyme_inducer_status",
            "amiodarone_status"
        ]
        self.col_names = col_names

        self.target_name = 'Therapeutic_Dose_of_Warfarin'

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

    def get_features(self, data):
        return data[self.col_names]

    def get_targets(self, data):
        return data[[self.target_name]]


class FDBaseline(DosageBaseline):
    """
    Implements fixed dosage baseline.
    Regardless of input, a fixed dosage is returned
    for each instance in the batch.
    """

    def __init__(self, dosage):
        self.dosage = dosage
        self.target_name = 'Therapeutic_Dose_of_Warfarin'

    def get_features(self, data):
        return data.drop(self.target_name, axis=1)  # Remove target from data

    def get_targets(self, data):
        return data[[self.target_name]]

    def predict(self, X):
        N, D = X.shape
        Y = np.ones((N, 1))
        return Y

    
class linearUCB(DosageBaseline):
    """
    Implements linear UCB, as seen in 
    http://john-maxwell.com/post/2017-03-17/
    """
    def __init__(self, num_arms, alpha=7):
        col_names = [
            "Age",
            "Height_(cm)",
            "Weight_(kg)",
            "VKORC1_A/G",
            "VKORC1_A/A",
            "VKORC1_nan",
            "Cyp2C9_*1/*2",
            "Cyp2C9_*1/*3",
            "Cyp2C9_*2/*2",
            "Cyp2C9_*2/*3",
            "Cyp2C9_*3/*3",
            "Cyp2C9_nan",
            "Race_Asian",
            "Race_Black_or_African_American",
            "Race_nan",
            "enzyme_inducer_status",
            "amiodarone_status",
        ]
        self.d = len(col_names)
        self.col_names = col_names
        self.target_name = 'Therapeutic_Dose_of_Warfarin'
        self.num_arms = num_arms
        self.alpha = alpha
        
        self.A = np.zeros((num_arms, self.d, self.d))
        for i in range(num_arms):
            self.A[i] = np.eye(self.d, self.d)
        self.b = np.zeros((num_arms, self.d))
            
        
    def get_features(self, data):
        return data[self.col_names]
    
    def get_targets(self):
        return data[self.target_name]
    
    def evaluate(self, X, target):
        num_incorrect = 0
        for j in range(X.shape[0]):
            x_ta = X[j]
            max_payoff = -float("inf")
            best_arm = 0
            for arm in range(self.num_arms):
                A_inv = np.linalg.inv(self.A[arm])
                theta = np.dot(A_inv, self.b[arm])
                payoff = np.dot(x_ta.T, theta) + self.alpha*(np.dot(np.dot(x_ta.T, A_inv), x_ta))**0.5

                if payoff > max_payoff:
                    max_payoff = payoff
                    best_arm = arm
                elif payoff == max_payoff:
                    best_arm = np.random.choice([best_arm, arm])
            if best_arm != target[j]:
                num_incorrect += 1
        return float(num_incorrect)/target.shape[0]
    
    def train(self, X, target):
        indices = range(X.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        target_shuffled = target[indices]
        regret = np.zeros((target.shape[0] + 1,))
        total_regret = 0
        incorrect_over_time = []
        
        for i in range(X.shape[0]):
            x_ta = X[i]
            max_payoff = -float("inf")
            best_arm = 0
            for j in range(self.num_arms):
                A_inv = np.linalg.inv(self.A[j])
                theta = np.dot(A_inv, self.b[j])
                payoff = np.dot(x_ta.T, theta) + self.alpha*(np.dot(np.dot(x_ta.T, A_inv), x_ta))**0.5
                
                if payoff > max_payoff:
                    max_payoff = payoff
                    best_arm = j
                elif payoff == max_payoff:
                    best_arm = np.random.choice([best_arm, j])
                    
            reward = 0
            if best_arm != target[i]:
                reward = -1
                total_regret += 1
            
            regret[i+1] = total_regret
            self.A[best_arm] += np.dot(x_ta.reshape((self.d, 1)), x_ta.T.reshape((1, self.d)))
            self.b[best_arm] += reward*x_ta
            
            ## evaluate
            if i%100 == 0:
                incorrect_over_time.append(self.evaluate(X, target))

        incorrect_over_time.append(self.evaluate(X, target))
        return regret, incorrect_over_time
            
            
            
            
                
            
                
        
        
        
        
    
    
    