from typing import List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import copy

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10

bestw=[]
besth=[]
minloss=100
maxacc=-1

def softmax(z):
    return np.exp(z) / (np.sum(np.exp(z), axis=0, keepdims=True))


def relu(input):
    relu = input.copy()
    relu[relu < 0] = 0  
    return relu


def reLu_dash(der):
    reLu_dash = der.copy()
    reLu_dash[reLu_dash <= 0] = 0
    reLu_dash[reLu_dash > 0] = 1
    return reLu_dash


def unpack(weightsAndBiases, hidden_num, H_Layers):
    Ws = []
    init = 0
    end = NUM_INPUT * hidden_num
    W = weightsAndBiases[init:end]
    Ws.append(W)

    for i in range(H_Layers - 1):
        init = end
        end = end + hidden_num * hidden_num
        W = weightsAndBiases[init:end]
        Ws.append(W)

    init = end
    end = end + hidden_num * NUM_OUTPUT
    W = weightsAndBiases[init:end]
    Ws.append(W)

    Ws[0] = np.array(Ws[0]).reshape(hidden_num, NUM_INPUT)
    for i in range(1, H_Layers):
        Ws[i] = Ws[i].reshape(hidden_num, hidden_num)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, hidden_num)

    # bias terms
    bias = []
    init = end
    end = end + hidden_num
    b = weightsAndBiases[init:end]
    bias.append(b)

    for i in range(H_Layers - 1):
        init = end
        end = end + hidden_num
        b = weightsAndBiases[init:end]
        bias.append(b)

    init = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[init:end]
    bias.append(b)

    return Ws, bias


def forward_prop(X, Y, weightsAndBiases, hidden_num, H_Layers):
    Hs = []
    Zs = []

    Ws, bs = unpack(weightsAndBiases, hidden_num, H_Layers)
    h = X

    for i in range(H_Layers):
        b = bs[i].reshape(-1, 1)
        Z = np.dot(Ws[i], h) + b
        Zs.append(Z)
        h = relu(Z)
        Hs.append(h)

    Zs.append(np.dot(Ws[-1], Hs[-1]) + bs[-1].reshape(-1, 1))

    yhat = softmax(np.dot(Ws[-1], Hs[-1]) + bs[-1].reshape(-1, 1))

    loss = np.sum(np.log(yhat) * Y)

    loss = (-1 / Y.shape[1]) * loss

    return loss, Zs, Hs, yhat

def updateparas(W, B, gradW, gradB, epsilon, alpha, trainY):
    for i in range(len(W)):
        W[i] = W[i] - (epsilon * gradW[i]) + (alpha * W[i] / trainY.shape[1])
        B[i] = B[i] - (epsilon * gradB[i])
    return W, B

def gradCE(X, Y, weightsAndBiases, hidden_num, H_Layers):
    loss, zs, hs, yhat = forward_prop(X, Y, weightsAndBiases, hidden_num, H_Layers)
    dJ_dWs = []  # List of gradients wrt weights
    dJ_dbs: list[Optional[Any]] = []  # List of gradients wrt biases

    Ws, bs = unpack(weightsAndBiases, hidden_num, H_Layers)
    G = yhat - Y

    for i in range(H_Layers, -1, -1):
        # Finding grad wrt b
        if i != H_Layers: #Finding dH/dz iteratively
            dh_dzs = reLu_dash(zs[i])
            G = dh_dzs * G

        dj_db_term = np.sum(G, axis=1) / Y.shape[1]
        dJ_dbs.append(dj_db_term)
        # Finding grad wrt w
        if i == 0:
            layer1 = np.dot(G, X.T) / Y.shape[1]
            dJ_dWs.append(layer1)

        else:
            dJ_dWs.append(np.dot(G, hs[i - 1].T) / Y.shape[1])

        G = np.dot(Ws[i].T, G)

    dJ_dbs.reverse()  
    dJ_dWs.reverse()  
    allGradientsAsVector=np.hstack([dJ_dW.flatten() for dJ_dW in dJ_dWs] + [dJ_db.flatten() for dJ_db in dJ_dbs]) #Flatten the gradients matrix
    
    return allGradientsAsVector


def findacc(yhat, y):
    yhat = yhat.T
    y = y.T
    Yhat = np.argmax(yhat, 1)
    Y = np.argmax(y, 1)
    acc = 100 * np.sum(Y == Yhat) / y.shape[0]
    return acc


def train(trainX, trainY, weightsAndBiases, hidden_num, H_Layers, epsilon, alpha):

    bp = gradCE(trainX, trainY, weightsAndBiases, hidden_num, H_Layers)
    gradW, gradB = unpack(bp, hidden_num, H_Layers)
    W, B = unpack(weightsAndBiases, hidden_num, H_Layers)
    W, B = updateparas(W, B, gradW, gradB, epsilon, alpha, trainY)

    weightsAndBiases = np.hstack([w.flatten() for w in W] + [b.flatten() for b in B])

    return weightsAndBiases


def sgd(train_X, train_Y, epochs, batch_size, weightsAndBiases, hidden_num, H_Layers, learning_rate, alpha,
        valid_X, valid_Y):
    global maxacc, minloss, besth, bestw
    for epoch in range(epochs):
        print("Epoch no.", epoch)
        N_batches = int((len(train_X.T) / batch_size))
        init = 0
        end = batch_size
        for i in range(N_batches):
            mini_batch = train_X[:, init:end]

            y_mini_batch = train_Y[:, init:end]

            weightsAndBiases = train(mini_batch, y_mini_batch, weightsAndBiases, hidden_num, H_Layers,
                                                 learning_rate, alpha)
            init = end
            end = end + batch_size

        loss, zs, hs, yhat = forward_prop(valid_X, valid_Y, weightsAndBiases, hidden_num, H_Layers)
        acc = findacc(yhat, valid_Y)
        
        if acc>maxacc:
            maxacc=acc
            minloss=loss
            besth = [epochs, batch_size, hidden_num, H_Layers, learning_rate, alpha]
            bestw=copy.deepcopy(weightsAndBiases)

        print("Loss: ", loss, "Accuracy: ", acc)

    return weightsAndBiases


def findBestHyperparameters(trainX, trainY, testX, testY):

    h_layers_list = [3]
    hidden_num_list = [81]
    mini_batch_size_list = [16] 
    epsilon_list = [0.005] 
    epochs_list = [70]
    alpha_list = [0.000001] 


    randorder = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, randorder]
    trainY = trainY[:, randorder]

    index_values = np.random.permutation(trainX.shape[1])
    train_X = trainX[:, index_values[:int(trainX.shape[1] * 0.8)]]
    valid_X = trainX[:, index_values[int(trainX.shape[1] * 0.8):]]
    train_Y = trainY[:, index_values[:int(trainX.shape[1] * 0.8)]]
    valid_Y = trainY[:, index_values[int(trainX.shape[1] * 0.8):]]


    for h_layers in h_layers_list:
        for hidden_num in hidden_num_list:
            for epochs in epochs_list:
                for batch_size in mini_batch_size_list:
                    for learning_rate in epsilon_list:
                        for alpha in alpha_list:

                            print("Hidden Layers: ", h_layers, "\nNeurons in each layer: ", hidden_num, "\nBatch_size=",
                                  batch_size)
                            print("Learning Rate: ", learning_rate, "\nAlpha: ", alpha, "\nEpochs: ", epochs)

                            weightsAndBiases = initWeightsAndBiases(hidden_num, h_layers)

                            weightsAndBiases = sgd(train_X, train_Y, epochs, batch_size,
                                                                    weightsAndBiases, hidden_num, h_layers,
                                                                    learning_rate, alpha, valid_X, valid_Y)

                            loss, hs, zs, yhat = forward_prop(valid_X, valid_Y, weightsAndBiases, hidden_num,
                                                                h_layers)

    best_epochs = besth[0]  # best number of epochs
    best_batch_size = besth[1]  # best number of batch_size
    best_hidden_num = besth[2]  # best number of hidden neurons
    best_h_layers = besth[3]  # best number of hidden layers
    best_learningrate = besth[4]  # best learning rate
    best_alpha = besth[5]  # best alpha

    weightsAndBiases = initWeightsAndBiases(best_hidden_num, best_h_layers)
    weightsAndBiases=copy.deepcopy(bestw)                                           
    loss, hs, zs, yhat = forward_prop(testX, testY, weightsAndBiases, best_hidden_num, best_h_layers)

    print("\nThe Best HyperParameters:  \nHidden Layers:", best_h_layers, "\nHidden Layer Neurons: ", best_hidden_num,
          "\nEpochs: ", best_epochs, "\nBatch size: ", best_batch_size)
    print("Learning rate: ", best_learningrate, "\nAlpha: ", best_alpha)
    print("Accuracy (validation data) :", maxacc)

    print("\nMin loss value: ", minloss)
    acc = findacc(yhat, testY)
    print("\nAccuracy on Test data: ", acc)
    print("\n")
    saveval(unpack(bestw, best_hidden_num, best_h_layers)[0],'weights.txt')
    saveval(unpack(bestw, best_hidden_num, best_h_layers)[1],'biases.txt')

    return weightsAndBiases, h_layers, hidden_num


def initWeightsAndBiases(hidden_num, H_Layers):
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2 * (np.random.random(size=(hidden_num, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(hidden_num)
    bs.append(b)

    for i in range(H_Layers - 1):
        W = 2 * (np.random.random(size=(hidden_num, hidden_num)) / hidden_num ** 0.5) - 1. / hidden_num ** 0.5
        Ws.append(W)
        b = 0.01 * np.ones(hidden_num)
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, hidden_num)) / hidden_num ** 0.5) - 1. / hidden_num ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

def show_W1 (W):
    Ws,bs = unpack(W, besth[2], NUM_HIDDEN_LAYERS)
    W = Ws[1]
    n = int(besth[2] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 9, 9]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()

def saveval(nested_list, name):
    with open(name, 'w') as file:
        for inner_list in nested_list:
            line = ','.join(map(str, inner_list)) + '\n'
            file.write(line)


if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255 #normalizing the data after loading it
    trainX = X_tr.T
    ytr = np.load("fashion_mnist_train_labels.npy")
    train_Y = ytr
    X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28 * 28)) / 255
    testX = X_te.T
    yte = np.load("fashion_mnist_test_labels.npy")
    test_Y = yte

    trainY = np.zeros((train_Y.size, train_Y.max() + 1))
    testY = np.zeros((test_Y.size, test_Y.max() + 1))
    trainY[np.arange(train_Y.size), train_Y] = 1
    testY[np.arange(test_Y.size), test_Y] = 1
    trainY = trainY.T
    testY = testY.T

    weightsAndBiases = initWeightsAndBiases(NUM_HIDDEN, NUM_HIDDEN_LAYERS)

    print("Gradient Check:")
    print(scipy.optimize.check_grad(
        lambda wab:
        forward_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab, NUM_HIDDEN, NUM_HIDDEN_LAYERS)[
            0],
        lambda wab: gradCE(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab, NUM_HIDDEN,
                              NUM_HIDDEN_LAYERS),
        weightsAndBiases))
    print("\n")
    wandb, H_Layers, hidden_num = findBestHyperparameters(trainX, trainY, testX, testY)
    change_order_idx = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, change_order_idx]
    trainY = trainY[:, change_order_idx]
    show_W1(wandb)
