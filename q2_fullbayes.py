

import argparse
import numpy as np
import pickle
from q1_fullbayes import input_dataset
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

def pass_parameters():
    # Pass the arguments on terminal.
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('dataset', help="Select a dataset(txt file) as input.")
    parser.add_argument('parameter', help="Model file contained parameters.")
    args = parser.parse_args()
    dataset_name = args.dataset # dataset_name = 'iris.txt.shuffled'
    parameters_file = args.parameter # parameters_file = 'Parameters_Full.txt'
    return dataset_name, parameters_file

def load_parameters(parameters_file):
    # load the parameter file
    with open(parameters_file, 'rb') as f:
        prior, mu, cov = pickle.load(f)
    return prior, mu, cov

def testing(X, num_class, prior, mu, cov):
    # Calculate 3 posterior probabilities of each instance and classify it into 1 of 3 classes.
    Y_pred = []
    for s in range(len(X)): # s: samples
        posterior = []
        for c in range(0, num_class): # c: classes
            # Calculate the posterior probabilities in class c: P(X|Y=C_c)*P(Y=C_c)
            post_class = multivariate_normal.pdf(X[s], mean=mu[c], cov=cov[c])*prior[c]
            posterior.append(post_class)
        # Choose the maximum posterior probabilities
        post_max = max(posterior)
        # Classify the predicted result
        for c in range(0, num_class):
            if post_max == posterior[c]: Y_pred.append(int(c+1))
    return np.array(Y_pred)


if __name__ == '__main__':
    # Input dataset and load parameters
    #dataset_name, parameters_file = pass_parameters()
    dataset_name = 'iris.txt.shuffled'
    parameters_file = 'Parameters_Full.pickle'
    test_X, test_Y, num_class = input_dataset(dataset_name)
    prior, mu, cov = load_parameters(parameters_file)

    # Test the model and return the predicted label
    Y_pred = testing(test_X, num_class, prior, mu, cov)
    # Present the evaluation
    print("Confusion Matrix: ")
    print(confusion_matrix(test_Y, Y_pred))
