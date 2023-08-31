import numpy as np
import math
import copy


#-------------------------------------------------------------------------Question 1-----------------------------------------------------------------------
def doCrossValidation (D, k, h):
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    accuracies = []
    for fold in range(k):
        # Get all indices for this fold
        testIdxs = idxs[fold,:]
        # Get all the other indices
        trainIdxs = np.array(set(allIdxs) - set(testIdxs)).flatten()
        # Train the model on the training data
        model = trainModel(D[trainIdxs], h)
        # Test the model on the testing data
        accuracies.append(testModel(model, D[testIdxs]))
    return np.mean(accuracies)


def doDoubleCrossValidation(D, k, H):
    l=k #Inner folds l=Outer folds k
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    accuracies = []
    for fold in range(k):
        # Get all indices for this fold
        testIdxs = idxs[fold, :]
        # Get all the other indices
        trainIdxs = np.array(set(allIdxs) - set(testIdxs)).flatten()
        acc_h_best=-1 #Initialise best accuracy
        for hps in H: #Loop through the set of hyperparameters
            allIdxsh = trainIdxs.copy() #The train indices of the outer fold become the indices of our new dataset for the inner fold
            idxsh = np.random.permutation(allIdxsh) # Randomly split dataset into l=k folds
            idxsh = idxsh.reshape(l, -1)
            acc_h=[]
            for f in range(l):
                testIdxsh = idxsh[f, :]  # Get all indices for this subfold
                trainIdxsh = np.array(set(allIdxsh) - set(testIdxsh)).flatten() # Get all the other indices for this subfold
                model_h = trainModel(D[trainIdxsh], hps)
                acc_h.append(testModel(model_h, D[testIdxsh]))
            acc_h_mean=np.mean(acc_h) #Find the average accuracy for current hyperparameter set
            if acc_h_mean>acc_h_best: #Check if current accuracy is better than old best accuracy
                acc_h_best=acc_h_mean
                h=copy.deepcopy(hps) #Save the best hyperparameter set for training on the entire data
        # Train the model on the training data
        model = trainModel(D[trainIdxs], h)
        # Test the model on the testing data
        accuracies.append(testModel(model, D[testIdxs]))
    return np.mean(accuracies)


#-------------------------------------------------------------------------Question 3-----------------------------------------------------------------------

#Creating a function for randomizing weights (which includes bias)
def linear_regression (X_tr):  
    w = np.random.randn(X_tr.shape[0])
    return w

#Created mini-batches of the training set
def mini_batch_generator(X_tr, ytr, mini_batch_size):
    m = X_tr.shape[1]
    permutation =list(np.random.permutation(m))
    X_tr = X_tr[:, permutation]
    ytr = ytr[permutation]
    inc = mini_batch_size
    mini_batches = []
    mini_batch_complete = math.floor(m/mini_batch_size)
    for i in range(0, mini_batch_complete):
        mini_batch_X_tr = X_tr[:, i*inc:(i+1)*inc]
        mini_batch_ytr = ytr[i*inc:(i+1)*inc]
        mini_batch = [mini_batch_X_tr, mini_batch_ytr]
        mini_batches.append(mini_batch)
        
    if m% mini_batch_size != 0:
        mini_batch_X_tr = X_tr[:, (inc*math.floor(m/inc)):]
        mini_batch_ytr = ytr[(inc*math.floor(m/inc)):]
        mini_batch = (mini_batch_X_tr, mini_batch_ytr)
        mini_batches.append(mini_batch)
    return mini_batches

#Using gradient of regularized loss function to update weights.
def update_params(w, mini_batch_X_tr, minibatch_ytr, alpha, learning_rate, mini_batch_size):
    #Excluding bias from the regularization term but updating it using mean squared error term
    w_withoutb = np.append(w[:-1], [0], axis = 0)
    w = w- 2*learning_rate*((mini_batch_X_tr.dot(np.dot(np.transpose(mini_batch_X_tr), w) - minibatch_ytr)/mini_batch_size) + alpha*w_withoutb)
    return w

#Train on training set, evaluate hyper parameters on validation set and finally testing on test set
def train_age_regressor():
    # Load data
    #Loading and reshaping the value of X_tr
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"),(-1, 48*48)) 
    #Appending 1's to the X_tr to find b (bias) for the training set while correcting it's shape
    Xtr = np.append(np.transpose(X_tr), [np.ones(np.transpose(X_tr).shape[1])], axis = 0)
    X_tr = Xtr[:, :4000]
    #Loading the y for training set
    y_tr = np.load("age_regression_ytr.npy")
    ytr = y_tr[:4000]
    X_val = Xtr[:, 4000:]
    y_val = y_tr[4000:]
    #Loading and reshaping the value of X_te    
    X_te = np.reshape(np.load("age_regression_Xte.npy"),(-1, 48*48))
    #Appending 1's to the X_te to find b (bias) for the testing set while correcting its shape
    X_te = np.append(np.transpose(X_te), [np.ones(np.transpose(X_te).shape[1])], axis = 0)
    #Loading the y for testing set
    yte = np.load("age_regression_yte.npy")
    #Getting w and b from the linear_regression() function
    #Creating lists to store cost of validation set, weights and hyper parameters
    cost = []
    weights = []
    h_params = []
    
    #Looping over several hyperparameters
    for mini_batch_size in [10, 50, 70, 80]:  #Best value = 50
        for learning_rate in [1e-3, 1e-4, 1e-5, 1e-6]:  #Best value = 1e-3
            for epochs in [500, 10, 20, 30]:  #Best value = 500
                for alpha in [1e-2, 1e-3, 1e-4, 1e-5]:  #Best value = 1e-2
                    w = linear_regression(X_tr)                
                    for epoch in range(epochs):
                        minibatches = mini_batch_generator(X_tr, ytr, mini_batch_size)
                        for minibatch in minibatches:
                            mini_batch_X_tr, mini_batch_ytr = minibatch
                            w = update_params(w, mini_batch_X_tr, mini_batch_ytr, alpha, learning_rate, mini_batch_size)
                            #Storing weights in the list initially created
                            weights.append(w)
                            h_params.append([mini_batch_size, learning_rate, epochs, alpha])
                            #Finding unregularized loss function on the validation set and storing it in the list (cost[]) initially created 
                            fmse_val = (np.mean(np.square(np.dot(np.transpose(X_val), w) - y_val)))/2
                            cost.append(fmse_val)
    #Getting index of the minimum value of fmse from the cost[] list
    ind = cost.index(min(cost))
    
    #Writing the hyper parameters of that index to a file
    file = open('h_params.txt','w')
    file.write(str(h_params[ind]))
    file.close()
    #Printing the best fmse value for validation set
    print("fmse_val = {}".format(min(cost)))
    
    # storing the weights for the same index
    file = open('weights.txt','w')
    for item in weights[ind]:
        file.write((str(item)+", "))
    file.close()
    #Printing the fsme test value AFTER THE TRAINING ENDS
    fmse_test = (np.mean(np.square(np.dot(np.transpose(X_te), weights[ind]) - yte)))/2
    print("fmse_test = {}".format(fmse_test))
                           
def main():
    train_age_regressor()
if __name__ == "__main__":main()
                            
                            