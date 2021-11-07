import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, mean_squared_error as mse
from sklearn.linear_model import LogisticRegression


class TwoStepTechnique(BaseEstimator, ABC):
    
    def __init__(self):
        self.classifier=None
    
    @abstractmethod
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray]:
        
        pass
    
    @abstractmethod
    def step2(self, X, n, p)->BaseEstimator:

        pass
    
    def fit(self, X, s):
        ## ADD YOUR CODE HERE
        return self
    
    def predict(self,X):
        ## ADD YOUR CODE HERE
        return None
    
    def predict_proba(self, X):
        ## ADD YOUR CODE HERE
        return None

class SEM(TwoStepTechnique):
    
    def __init__(self,
                 tol = 1.0e-10,
                 max_iter = 100,
                 spy_prop = 0.1,
                 l = 0.15,
                 classifier = LogisticRegression(),
                 seed=331
                ):
        
        super().__init__()
        
        # instantiate the parameters
        ## ADD YOUR CODE HERE
        
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray]:
        
        np.random.seed(self.seed)
        
        ## Split the dataset into P (Positive) and M (Mix of positives and negatives)
        n = len(X)
        
        P = np.copy(X[np.where(s==1)])
        M = np.copy(X[np.where(s==0)])
        
        # Select (randomly) the spies S
        len_p = len(P)
        index_set = np.arrange(len_p-1)
        percent_spies = 0.15
        Spy_size = int(np.around(len_p * percent_spies))
        Spy_index = np.random.choice(index_set, size=Spy_size, replace=False)
        Spies = P[Spy_index,:]
        
        # Update P and MS
        MS = np.concatenate((M,Spies))
        index_p = np.delete(np.arange(len_p-1),Spy_index)
        P = [P[i] for i in index_p]
        
        ### I-EM Algorithm

        # Train the classifier using P and MS:
        gnb = GaussianNB()
        label_ms = np.zeros(len(MS))
        label_p = np.ones(len(P))
        train = np.concatenate((MS,P))
        train_labels = np.concatenate((label_ms,label_p))
        model = gnb.fit(train, train_labels)

        # Save the model's score ''score_variation'' using model.score function
        score_variation = 1
        old = model.score(train, train_labels)
        new = 0
        
        # Initialize iterations to 0 and the score variation
        n_iter = 0
        
        
        #Loop while classifier parameters change, i.e. until the score variation is >= tolerance
        while score_variation >= self.tol and n_iter < self.max_iter:
            

            # Expectation step
            preds = gnb.predict(MS)
            
            # Create the new training set with the probabilistic labels (weights)
            train_labels = np.concatenate((preds,label_p))
            
            # Maximization step
            model = gnb.fit(train, train_labels)
            
            # Update score variation and the old score
            new = model.score(train, train_labels)
            score_variation = np.absolute(new-old)
            old = new
            n_iter += 1
            
        # Print the number of iterations as sanity check
        print("Number of iterations first step:", n_iter)
        
        # Select the threshold t such that l% of spies' probabilities to be positive is belot t

        
        # Create N and U

        
        # Return P, N, U
        #return P, N, U
        
    
    def step2(self, X, P, N, U)->BaseEstimator:
        np.random.seed(self.seed)
        
        # Assign every document in the positive set pos the fixed class label 1

        
        # Assign every document in the likely negative set N the initial class label 0

        
        ###I-EM Algorithm


        # Train classifier using M and P:

        
        # Compute the metrics for classifier f_i in delta_i to select the best classifier

        

        
        # Initialize iterations to 0, the score variation, and whether the best classifier has been selected or not.

        
        # Loop until the variation is > than the tolerance
        #while score_variation >= self.tol and n_iter < self.max_iter:

            
            # Update probabilities

            
            
            # Create the new training set with the probabilistic labels (weights)

            
            
            # Maximization step

            
            # Update parameter variation

            
            # Select the best classifier classifier: (final_classifier)

            

        #print("Number of iterations second step:", n_iter)
        
        
        return self.final_classifier

    
    
#--------------------#--------------------#--------------------#--------------------    
#-------------------- Second PU Learning Method #--------------------


    
class ModifiedLogisticRegression(BaseEstimator):

    def __init__(self,
                 tol = 1.0e-10,
                 max_iter = 100,
                 l_rate = 0.001,
                 c = 0,
                 seed = 331):
        
        #instantiate the parameters
        ## ADD YOUR CODE HERE
        super().__init__()

    def log_likelihood(self, x, y):
        #If you use the gradient ascent technique, fill in this part with the log_likelihood function
        #If you prefer to use a different technique, you can leave this empty
        
        
        return
        
    def parameters_update(self, x, y):
        #If you use the gradient ascent technique, fill in this part with the parameter update (both w and b)
        #If you prefer to use a different technique, you can leave this empty
        
        
        return 
    
    def fit(self, X, s):
        np.random.seed(self.seed)
        
        # Initialize w and b

        
        # Inizialize the score (log_likelihood), the number of iterations and the score variation.

        
        #loop until the score variation is lower than tolerance or max_iter is reached
        # while score_variation >= self.tol and n_iter < self.max_iter:
            
            # Update the parameters

            
            # Compute log_likelihood

            
            # Update scores

            
        return self    
    
    def estimate_c(self):
        # Estimate the parameter c from b
        ## ADD YOUR CODE HERE
        return
    
    def predict(self,X):
        ## ADD YOUR CODE HERE
        return None
    
    def predict_proba(self, X):
        ## ADD YOUR CODE HERE
        return None
    
    
    
    
