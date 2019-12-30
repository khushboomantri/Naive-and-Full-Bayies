

import argparse
import numpy as np
import pickle

def pass_parameters():
    # Pass the arguments on terminal.
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('dataset', help="Select a dataset(txt file) as input.")
    args = parser.parse_args()
    dataset_name = args.dataset # dataset_name = 'iris.txt.shuffled'
    return dataset_name

def input_dataset(file_name):
    # Read dataset
    train_data = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.strip():
                train_data.append(line.strip())
    # Split the dataset into samples and labels.
    samples, labels = [], []
    for data in train_data:
        x = data.split(',')[:-1]
        samples.append(x)
        y = data.split(',')[-1]
        labels.append(y)
    # Detect the number of classes and replace labels
    label_dict = dict(zip(labels, labels))
    num_class = len(label_dict)
    type_class = [key for key in label_dict.keys()]
    for i in range(len(labels)):
        for j in range(num_class):
            if labels[i] == type_class[j]:
                labels[i] = int(j+1)
    return np.array(samples).astype('float32'), np.array(labels), num_class

def bayes_Classifier(X, Y, num_class, parameters_file='Parameters_Full'):
    # Separate the dataset into different classes
    D = [[] for i in range(0, num_class)]
    for i in range(Y.shape[0]):
        for j in range(len(D)):
            if Y[i] == int(j+1): D[j].append(X[i])
    # Compute the parameters
    prior, mu, cov = [], [], []
    for c in range(len(D)):
        D[c] = np.array(D[c])
        # Compute the prior probability of each class
        p_c = len(D[c])/len(X)
        prior.append(p_c)
        # Compute the mean of n features in c classes
        mu_c = np.mean(D[c], axis=0)
        mu.append(mu_c)
        # Compute the covariance of each feature in each class
        cov_c = np.cov(D[c].T)
        cov.append(cov_c)
    # Save the parameters in pickle file
    para_pickle = str(parameters_file+'.pickle')
    with open(para_pickle, 'wb') as f:
        pickle.dump([prior, mu, cov], f)
    # Save the parameters in text file
    para_txt = str(parameters_file+'.txt')
    open(para_txt,'w').close() # clear previous content
    with open(para_txt, 'a') as f:
        f.write("Prior probabilities of 3 classes = ")
        f.write("\n")
        np.savetxt(f, np.array(prior).reshape((1, len(prior))), fmt='%0.2f')
        f.write("\n")
        for i in range(len(mu)):
            f.write("Mean value of class {:.0f} = ".format(i+1))
            f.write("\n")
            np.savetxt(f, mu[i].reshape((1, len(mu[0]))), fmt='%0.2f')
            f.write("\n")
        for i in range(len(cov)):
            if i != 0: f.write("\n")
            f.write("Covariance of class {:.0f} = ".format(i+1))
            f.write("\n")
            for j in range(len(cov[0])):
                np.savetxt(f, cov[i][j].reshape((1, len(cov[0]))), fmt='%0.2f')
    return para_pickle


if __name__ == '__main__':
    # Split and input dataset
    #file_name = pass_parameters() # python q1_fullbayes.py iris.txt.shuffled
    file_name = 'iris.txt.shuffled'
    train_X, train_Y, num_class = input_dataset(file_name)

    # Train the bayes classifier and save its parameters
    para_pickle = bayes_Classifier(train_X, train_Y, num_class)
    print("Parameter file has been output.")
    print("Please use '", para_pickle, "' as the second argument in Q2.")
