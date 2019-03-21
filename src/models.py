import numpy as np
import pandas as pd
from sklearn import linear_model
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

        self.target_name = 'Therapeutic_Dose_of_Warfarin_binned'

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

        self.target_name = 'Therapeutic_Dose_of_Warfarin_binned'

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
        self.target_name = 'Therapeutic_Dose_of_Warfarin_binned'

    def get_features(self, data):
        return data.drop(self.target_name, axis=1)  # Remove target from data

    def get_targets(self, data):
        return data[[self.target_name]]

    def predict(self, X):
        N, D = X.shape
        Y = np.ones((N, 1))
        return Y


class LinearUCB(DosageModel):
    """
    Implements linear UCB, as seen in
    http://john-maxwell.com/post/2017-03-17/
    """
    def __init__(self, num_arms, alpha=7, reward_correct=0, reward_incorrect_1=-1, reward_incorrect_2=-1):
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
        self.target_name = 'Therapeutic_Dose_of_Warfarin_binned'
        self.num_arms = num_arms
        self.alpha = alpha
        self.regret = None
        self.total_regret = 0
        self.incorrect_over_time = []
        self.reward_correct = reward_correct
        self.reward_incorrect_1 = reward_incorrect_1
        self.reward_incorrect_2 = reward_incorrect_2

        self.A = np.zeros((num_arms, self.d, self.d))
        for i in range(num_arms):
            self.A[i] = np.eye(self.d, self.d)
        self.b = np.zeros((num_arms, self.d))


    def get_features(self, data):
        return data[self.col_names]

    def get_targets(self):
        return data[self.target_name]

    def evaluate_single_example(self, x_ta):
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
        return best_arm

    def evaluate(self, X, target):
        num_incorrect = 0
        for j in range(X.shape[0]):
            x_ta = X[j]
            best_arm = self.evaluate_single_example(x_ta)
            if best_arm != target[j]:
                num_incorrect += 1
        return float(num_incorrect)/target.shape[0]

    def train_predict_single_example(self, x_ta):
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

        return best_arm
    

    def update_parameters(self, x_ta, best_arm, target, t):
        reward = self.reward_correct
        if np.absolute(best_arm - target) == 2:
            reward = self.reward_incorrect_2
        elif np.absolute(best_arm - target) == 1:
            reward = self.reward_incorrect_1
            
        self.total_regret -= reward
        self.regret[t+1] = self.total_regret
        self.A[best_arm] += np.dot(x_ta.reshape((self.d, 1)), x_ta.T.reshape((1, self.d)))
        self.b[best_arm] += reward*x_ta


    def setup_training(self, N):
        self.regret = np.zeros((N + 1,))
        self.total_regret = 0
        self.incorrect_over_time = []


    def train(self, X, target):
        indices = list(range(X.shape[0]))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        target_shuffled = target[indices]

        self.setup_training(target_shuffled.shape[0])

        for i in range(X_shuffled.shape[0]):
            ## evaluate
            if i%100 == 0:
                self.incorrect_over_time.append(self.evaluate(X_shuffled, target_shuffled))
                
            X_ta = X_shuffled[i]
            pi_t = self.train_predict_single_example(X_ta)
            self.update_parameters(X_ta, pi_t, target_shuffled[i], i)

        self.incorrect_over_time.append(self.evaluate(X_shuffled, target_shuffled))
        return self.regret, self.incorrect_over_time

    
class Lasso(DosageModel):
    """
    Implements lasso bandit, as seen in
    http://web.stanford.edu/~bayati/papers/lassoBandit.pdf
    """
    def __init__(self, num_arms, lambda1, lambda2, h, q, reward_correct=0, reward_incorrect_1=-1, reward_incorrect_2=-1):
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
        self.target_name = 'Therapeutic_Dose_of_Warfarin_binned'
        self.num_arms = num_arms
        self.lambda1 = lambda1
        self.lambda2_0 = lambda2
        self.lambda2_t = lambda2
        self.h = h
        self.q = q
        self.regret = None
        self.total_regret = 0
        self.incorrect_over_time = []
        self.Y = None
        self.lasso_l1 = None
        self.reward_correct = reward_correct
        self.reward_incorrect_1 = reward_incorrect_1
        self.reward_incorrect_2 = reward_incorrect_2


    def construct_forced_samples(self, N):
        """
        Constructs forced samples for each arm.
        T[i, t] == 1 if arm i is forced sampled for time / sample t,
        otherwise, T[i, t] == 0.

        Arguments:
            N (int): Total number of timesteps / samples
        """
        K, h, q = self.num_arms, self.h, self.q
        for i in range(1, K+1):
            # n = {0, 1, 2, 3, 4, ...}
            n_upper = int(np.log(N)/np.log(2))
            n_idxs = np.arange(n_upper)
            # j = {q*i, q*i + 1, q*i + 2, ..., q*i}
            j_idxs = np.arange(q*(i-1)+1, q*(i)+1)
            # sampled indices = (2^n - 1)*Kq + j
            idxs = []
            for n in n_idxs:
                for j in j_idxs:
                    idxs.append((np.power(2, n)-1)*(K*q) + j)

            idxs = np.array(idxs) - 1
            # Remove out of bound indices
            idxs = idxs[(idxs < N) & (idxs >= 0)]

            self.T[i-1, idxs] = 1


    def get_features(self, data):
        return data[self.col_names]

    def get_targets(self):
        return data[self.target_name]

    def evaluate_single_example(self, X_t):
        K, h = self.num_arms, self.h
        approx_rewards_T = np.zeros((K,))
        for i in range(K):
            approx_rewards_T[i] = np.dot(X_t, self.beta_T[i])

        # Define reward threshold for which arm will be included in K_hat
        reward_thresh = np.max(approx_rewards_T) - h/2.0
        # Get arms for which approx rewards under T are >= reward thresh
        K_hat = np.arange(K)[approx_rewards_T >= reward_thresh]

        pi_t = 0
        max_reward = -float('inf')
        for idx in range(len(K_hat)):
            i = K_hat[idx]
            aprrox_reward = np.dot(X_t, self.beta_S[i])

            if aprrox_reward > max_reward:
                max_reward = aprrox_reward
                pi_t = i

        return pi_t

    def evaluate(self, X, target):
        num_incorrect = 0
        N, D = X.shape
        K, h = self.num_arms, self.h
        
        for t in range(X.shape[0]):
            X_t = X[t]
            pi_t = self.evaluate_single_example(X_t)
                
            if pi_t != target[t]:
                num_incorrect += 1
        
        return float(num_incorrect)/target.shape[0]

    def train_predict_single_example(self, X_t, t, X_shuffled):  
        N, D = X_shuffled.shape
        K, h = self.num_arms, self.h

        pi_t = None; # Chosen arm / action
        # Check if current sample has been sampled by force by any arm
        if np.sum(self.T[:,t]) > 0:
            # Assign chosen arm for current sample pi_t, arm that forced sampled
            pi_t = np.argmax(self.T[:, t])
        else:
            # Keep track of estimated forced rewards for each arm
            approx_rewards_T = np.zeros((K,))
            for i in range(K):
                # Get indices for forced samples of arm i up to sample t
                idxs = np.arange(N, dtype=np.int32)[self.T[i].astype(bool)][:t]
                self.lasso_l1.fit(X_shuffled[idxs], self.Y[idxs])
                # Get fitted beta parameters
                self.beta_T[i] = self.lasso_l1.coef_
                approx_rewards_T[i] = np.dot(X_t, self.beta_T[i])

            # Define reward threshold for which arm will be included in K_hat
            reward_thresh = np.max(approx_rewards_T) - h/2.0
            # Get arms for which approx rewards under T are >= reward thresh
            K_hat = np.arange(K)[approx_rewards_T >= reward_thresh]

            pi_t = 0
            max_reward = -float('inf')
            for idx in range(len(K_hat)):
                i = K_hat[idx]
                # Create new lasso estimator for set S on new lambda 2
                lasso_l2 = linear_model.Lasso(self.lambda2_t / 2.0, fit_intercept=False, max_iter=7000)
                # Get indices for free samples of arm i up to sample t
                idxs = np.arange(N, dtype=np.int32)[self.S[i].astype(bool)][:t]
                if len(idxs) > 0:
                    lasso_l2.fit(X_shuffled[idxs], self.Y[idxs])
                    # Get fitted beta parameters
                    self.beta_S[i] = lasso_l2.coef_

                aprrox_reward = np.dot(X_t, self.beta_S[i])

                if aprrox_reward > max_reward:
                    max_reward = aprrox_reward
                    pi_t = i
        return pi_t


    def update_parameters(self, pi_t, target, t):
        D = self.d
        self.S[pi_t, t] = 1 # Update total free samples
        # Update lambda 2
        self.lambda2_t = self.lambda2_0 * np.sqrt((np.log(t+1) + np.log(D))/(t+1))  # added 1 to t because t vals supposed to be 1 indexed
        # Record reward of taking pi_t action
        reward = self.reward_correct
        if np.absolute(pi_t - target) == 2:
            reward = self.reward_incorrect_2
        elif np.absolute(pi_t - target) == 1:
            reward = self.reward_incorrect_1

        # Update targets with actual target at t
        self.Y[t] = reward

        self.total_regret -= reward
        self.regret[t+1] = self.total_regret


    def setup_training(self, N, K):
        D = self.d
        self.total_regret = 0
        self.regret = np.zeros((N + 1,))
        self.incorrect_over_time = []
        
        # Forced samples for each arm
        # T[i, t] == 1 if arm i is forced sampled at time / sample t,
        self.T = np.zeros((K, N))
        # Free samples for each arm
        # S[i, t] == 1 if arm i is free sampled at time / sample t,
        self.S = np.zeros((K, N))
        # Beta parameters for each arm under forced set T
        self.beta_T = np.zeros((K, D))
        # Beta parameters for each arm under free set S
        self.beta_S = np.zeros((K, D))
        # Targets
        self.Y = np.zeros((N,))

        # Define lasso estimator for forced set T on lambda 1
        self.lasso_l1 = linear_model.Lasso(self.lambda1 / 2.0, fit_intercept=False, max_iter=7000)

        # Populate forced samples for each arm
        self.construct_forced_samples(N)

        
    def train(self, X, target):
        N, D = X.shape
        K, h = self.num_arms, self.h

        indices = list(range(X.shape[0]))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        target_shuffled = target[indices]

        self.setup_training(N, K)

        for t in range(N):
            if t%100 == 0:
                self.incorrect_over_time.append(self.evaluate(X_shuffled, target_shuffled))
                
            X_t = X_shuffled[t]
            pi_t = self.train_predict_single_example(X_t, t, X_shuffled)
            self.update_parameters(pi_t, target_shuffled[t], t)

        self.incorrect_over_time.append(self.evaluate(X_shuffled, target_shuffled))
        return self.regret, self.incorrect_over_time


class Ensemble(DosageModel):
    """
    Ensembles the three baseline & two other models together
    """
    def __init__(self, num_arms, reward_correct=0, reward_incorrect_1=-1, reward_incorrect_2=-1):
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
        CDA_col_names = [
            "Age",
            "Height_(cm)",
            "Weight_(kg)",
            "Race_Asian",
            "Race_Black_or_African_American",
            "Race_nan",
            "enzyme_inducer_status",
            "amiodarone_status"
        ]
        self.d = len(col_names)
        self.col_names = col_names
        self.CDA_col_names = CDA_col_names
        self.target_name = 'Therapeutic_Dose_of_Warfarin_binned'
        self.num_arms = num_arms
        self.regret = None
        self.total_regret = 0
        self.incorrect_over_time = []

        self.FDBaseline = FDBaseline(1)
        self.PDABaseline = PDABaseline()
        self.CDABaseline = CDABaseline()
        self.LinearUCB = LinearUCB(num_arms=3, alpha=0.5)
        self.Lasso = Lasso(num_arms=3, lambda1=0.05, lambda2=0.05, h=5, q=1)
        self.num_models = 5

        self.reward_correct = reward_correct
        self.reward_incorrect_1 = reward_incorrect_1
        self.reward_incorrect_2 = reward_incorrect_2

    def get_features(self, data):
        return data[self.col_names], data[self.CDA_col_names]

    def get_targets(self):
        return data[self.target_name]

    def evaluate(self, X, X_CDA, target):
        num_incorrect = 0
        N, D = X.shape
        K = self.num_arms
        
        for t in range(X.shape[0]):
            X_t = X[t]
            X_t_CDA = X_CDA[t]

            pi_t = np.zeros((self.num_models,))
            pi_t[0] = self.FDBaseline.predict(X_t.reshape((1, X_t.shape[0])))[0]
            pi_t[1] = self.PDABaseline.predict(X_t.reshape((1, X_t.shape[0])))[0]
            pi_t[2] = self.CDABaseline.predict(X_t_CDA.reshape((1, X_t_CDA.shape[0])))[0]
            pi_t[3] = self.LinearUCB.evaluate_single_example(X_t)
            pi_t[4] = self.Lasso.evaluate_single_example(X_t)

            counts = np.bincount(pi_t.astype('int64'))
            winners = list(np.argwhere(counts == np.amax(counts)))
            best_arm = winners[0]
            if len(winners) > 1:
                if 1 in winners:
                    best_arm = 1
                else:
                    best_arm = min(winners)
                
            if best_arm != target[t]:
                num_incorrect += 1
        
        return float(num_incorrect)/target.shape[0]


    def train(self, X, X_CDA, target):

        N, D = X.shape
        K = self.num_arms

        indices = list(range(X.shape[0]))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        X_shuffled_CDA = X_CDA[indices]
        target_shuffled = target[indices]

        self.LinearUCB.setup_training(N)
        self.Lasso.setup_training(N, K)

        self.total_regret = 0
        self.regret = np.zeros((N + 1,))
        self.incorrect_over_time = []

        for t in range(N):
            if t%100 == 0:
                self.incorrect_over_time.append(self.evaluate(X_shuffled, X_shuffled_CDA, target_shuffled))
                
            X_t = X_shuffled[t]
            X_t_CDA = X_shuffled_CDA[t]

            pi_t = np.zeros((self.num_models,))
            pi_t[0] = self.FDBaseline.predict(X_t.reshape((1, X_t.shape[0])))[0]
            pi_t[1] = self.PDABaseline.predict(X_t.reshape((1, X_t.shape[0])))[0]
            pi_t[2] = self.CDABaseline.predict(X_t_CDA.reshape((1, X_t_CDA.shape[0])))[0]
            pi_t[3] = self.LinearUCB.train_predict_single_example(X_t)
            pi_t[4] = self.Lasso.train_predict_single_example(X_t, t, X_shuffled)

            counts = np.bincount(pi_t.astype('int64'))
            winners = list(np.argwhere(counts == np.amax(counts)))
            best_arm = winners[0]
            if len(winners) > 1:
                if 1 in winners:
                    best_arm = 1
                else:
                    best_arm = min(winners)
            
            self.LinearUCB.update_parameters(X_t, best_arm, target[t], t)
            self.Lasso.update_parameters(best_arm, target[t], t)

            reward = self.reward_correct
            if np.absolute(best_arm - target[t]) == 2:
                reward = self.reward_incorrect_2
            elif np.absolute(best_arm - target[t]) == 1:
                reward = self.reward_incorrect_1

            self.total_regret -= reward
            self.regret[t+1] = self.total_regret

        self.incorrect_over_time.append(self.evaluate(X_shuffled, X_shuffled_CDA, target_shuffled))
        return self.regret, self.incorrect_over_time



    
    
    


