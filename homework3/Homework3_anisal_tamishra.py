import numpy as np

# Creating a function to calculate the cross-entropy 
def Fce(X, y, w, b, alpha): 
    exp_z = np.exp(np.dot(np.transpose(X), w) + b)
    sum_expz = np.reshape(np.sum(exp_z, axis=1), (-1, 1))
    y_hat = exp_z / sum_expz
    Fce = -np.sum(y * np.log(y_hat)) / X.shape[1]
    return [Fce, y_hat]


#creating a function to calculate the stochastic gradient descent
def sgd(epochs, learning_rate, alpha, mini_batch_size, X_train, Y_train, w, b):
    for i in range(epochs):
        batch = int((len(np.transpose(X_train)) / mini_batch_size)) #Number of batches
        #Creating the start and end points
        start = 0
        end = mini_batch_size
        #Looping over total batches
        for j in range(batch):
            mini_batch = X_train[:, start:end]

            y_mini_batch = Y_train[start:end, :]

            exp_z = np.exp(np.dot(np.transpose(mini_batch), w) + b)  # find exp_z
            sum_expz = np.reshape(np.sum(exp_z, axis=1), (-1, 1))  # find mean of exp_z
            y_hat = exp_z / sum_expz
            #Calculating gradients to update the parameters
            #Including the regularization just for weight and not bias
            grad_w = (np.dot(mini_batch, (y_hat - y_mini_batch))) / mini_batch.shape[1] + alpha * w
            grad_b = np.array([(-(np.sum((y_mini_batch - y_hat), 0)) / mini_batch.shape[1])])

            #Updating parameters
            w_values = w - (np.dot(learning_rate, grad_w))
            b_values = b - (np.dot(learning_rate, grad_b))

            start = end
            end = end + mini_batch_size
            w = w_values
            b = b_values
            
        fce_each_epoch, _ = Fce(X_train, Y_train, w, b, alpha)
        fce_each_epoch += (alpha / 2) * (np.sum(np.dot(np.transpose(w), w)))
    return [fce_each_epoch, w, b]


def softmax():
    # Loading the dataset
    X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28 * 28))
    ytr = np.load("fashion_mnist_train_labels.npy")
    X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28 * 28))
    yte = np.load("fashion_mnist_test_labels.npy")

    Xtr = np.transpose(X_tr)


    index_values = np.random.permutation(Xtr.shape[1])
    #Splitting the train dataset into training and validation set
    X_train = Xtr[:, index_values[:int(Xtr.shape[1] * (80/100))]]
    X_vset = Xtr[:, index_values[int(Xtr.shape[1] * (80/100)):]]

    Y_train = ytr[index_values[:int(Xtr.shape[1] * (80/100))]]
    Y_vset = ytr[index_values[int(Xtr.shape[1] * (80/100)):]]

    Ytrain = np.zeros((Y_train.size, Y_train.max() + 1))
    Yvalid = np.zeros((Y_vset.size, Y_vset.max() + 1))
    Ytrain[np.arange(Y_train.size), Y_train] = 1
    Yvalid[np.arange(Y_vset.size), Y_vset] = 1

    #Structuring the test set to get correct shapes
    Xtest = X_te.T
    Ytest = np.zeros((yte.size, yte.max() + 1))
    Ytest[np.arange(yte.size), yte] = 1

    # Random initialization of w and zeros for b
    w = np.random.randn(int(Xtr.shape[0]))
    w = np.atleast_2d(w).T
    b = np.random.randn(10)

    #Considering 4 values of each hyperparameter
    epochs = [60, 75, 100, 200] 
    learning_rate = [2e-6, 4e-6, 5e-6, 6e-6] 
    alpha = [0.5, 0.8, 1.0, 2.0] 
    mini_batch_size = [128, 256, 512, 1024] 

    #Initializing min Cross Entropy to infinity
    Fce_min = np.inf

    #Looping over all values of hyperparameters
    for epoch in epochs:
        for ep in learning_rate:
            for al in alpha:
                for m in mini_batch_size:
                    Fce_, w, b = sgd(epoch, ep, al, m, X_train, Ytrain, w, b)
                    Fce_valid, _ = Fce(X_vset, Yvalid, w, b, al)
                    if Fce_valid < Fce_min:
                        hyper_parameters = [epoch, ep, al, m]
                        Fce_min = Fce_valid
                        
    #Getting the values of minimum cross entropy for train and test sets
    print("Minimum Fce for validation set= ", Fce_min)
    epoch_best = hyper_parameters[0]
    learning_rate_best = hyper_parameters[1]
    alpha_best = hyper_parameters[2]
    minibatch_best = hyper_parameters[3]
    
    # Randomly initializing of weights and bias
    w = np.random.randn(int(Xtr.shape[0]))
    w = np.atleast_2d(w).T
    b = np.random.randn(10)
    Fce_train, weights, bias = sgd(epoch_best, learning_rate_best, alpha_best, minibatch_best, X_train,
                                   Ytrain, w, b)
    Fce_test, Y_hat = Fce(Xtest, Ytest, weights, bias, alpha_best)
    Y_hat = np.argmax(Y_hat, 1)
    accuracy = 100 * np.sum(yte == Y_hat) / X_te.shape[0]
    print("Cross Entropy on the test set with best hyperparameters: ", Fce_test)
    print("Best hyper parameters:  Learning rate:", learning_rate_best, "epochs: ", epoch_best, " mini_batch_size: ", minibatch_best, "alpha: ", alpha_best)
    print("accuracy: ", accuracy)


def main():
    softmax()
if __name__ == "__main__":main()