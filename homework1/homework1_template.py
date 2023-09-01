import numpy as np
import os

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return (np.dot(A, B) - C)

def problem_1c (A, B, C):
    return ((A*B)+ np.transpose(C))

def problem_1d (x, y):
    return x.dot(y) #

def problem_1e (A, x):
    return np.dot(np.linalg.solve(A, x))

def problem_1f (A, i):
    A = A[i,:]
    A = A[::2]
    return np.sum(A) 

def problem_1g (A, c, d):
    A = A[np.nonzero(A<=d)]
    A = A[np.nonzero(c<=A)]
    return np.mean(A)

def problem_1h (A, k):
    u, v = np.linalg.eig(A)
    temp_mat = np.vstack((np.abs(u), v))
    temp_mat = np.transpose(temp_mat)
    final = sorted(temp_mat, key= lambda x:x[0], reverse=True)
    final = np.transpose(final)
    final = np.delete(final, 0, 0)
    return final[:,:k]

def problem_1i (x, k, m, s):
    z = np.ones((x.shape)[0])
    I = np.identity(np.shape(x)[0])
    N = np.empty(shape=((x.shape)[0], k))
    return np.transpose((np.random.multivariate_normal(x + m*z, s*I, size = k)))

def problem_1j (A):
    A = A[:, np.random.permutation(A.shape[1])]
    return A

def problem_1k (x):
    return (x - np.mean(x))/np.std(x)

def problem_1l (x, k):
    return np.repeat(np.reshape(x, (x.shape[0], 1)), k, axis = 1)

def problem_1m (X, Y):
    n=X.shape[1]
    m=Y.shape[1]
    X = X[:, :, np.newaxis]
    Y = Y[:, :, np.newaxis]
    X = np.repeat(X, m, axis=2)
    Y = np.repeat(Y, n, axis=2)
    X=np.swapaxes(X,1,2)
    Ans=X-Y
    return np.transpose(np.sum(Ans**2, axis=0))

def problem_1n (matrices):
    sum = 0
    nr1, nc1 = np.shape(matrices[0])
    for i in range(1, len(matrices)):
        nc2 = np.shape(matrices[i])[1]
        sum = sum + nr1*nc1*nc2
        nc1 = nc2
    return sum

def linear_regression (X_tr, y_tr):
    w = np.linalg.inv(X_tr.dot(np.transpose(X_tr))).dot(X_tr.dot(y_tr))
    b = w[-1]
    return w, b

def train_age_regressor():
    # Load data
    #Loading and reshaping the value of X_tr
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"),(-1, 48*48)) 
    #Appending 1's to the X_tr to find b (bias) for the training set while correcting it's shape
    X_tr = np.append(np.transpose(X_tr), [np.ones(np.transpose(X_tr).shape[1])], axis = 0)
    #Loading the y for training set
    ytr = np.load("age_regression_ytr.npy")
    
    #Loading and reshaping the value of X_te    
    X_te = np.reshape(np.load("age_regression_Xte.npy"),(-1, 48*48))
    #Appending 1's to the X_te to find b (bias) for the testing set while correcting it's shape
    X_te = np.append(np.transpose(X_te), [np.ones(np.transpose(X_te).shape[1])], axis = 0)
    #Loading the y for testing set
    yte = np.load("age_regression_yte.npy")
    
    #Getting w and b from the linear_regression() function
    w, b = linear_regression(X_tr, ytr)
    
    # Report fMSE cost on the training and testing data (separately)
    #Instead of using b explicitly in the function, I directly utilized the implicit value of b generated after X and w are multiplied
    fmse_train = np.mean(np.square(np.dot(np.transpose(X_tr), w) - ytr))/2
    fmse_test = np.mean(np.square(np.dot(np.transpose(X_te), w) - yte))/2
    print("The FSME for train dataset is : {}".format(fmse_train))
    print("The FSME for test dataset is : {}".format(fmse_test))


def main():
    train_age_regressor()
if __name__ == "__main__":main()