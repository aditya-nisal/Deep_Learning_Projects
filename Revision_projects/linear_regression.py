import numpy as np

def linear_regression (X_tr, y_tr):
    w = np.dot(np.linalg.inv(np.dot(X_tr, X_tr.T)), np.dot(X_tr, y_tr))
    b = w[-1]
    return w, b

def train_age_regressor ():
    
    # Training part
    # Load data. 
    X_tr = np.reshape(np.load("/home/aditya/Deep_Learning_Projects/homework1/age_regression_Xtr.npy"), (-1, 48*48)) # Reshaped x to a vector of 5000 images
    # X_tr = np.hstack((X_tr, np.ones((X_tr.shape[0],1) )))
    X_ones = np.ones((X_tr.shape[0], 1))
    X_tr = np.transpose(np.hstack((X_tr, X_ones)))
    ytr = np.load("/home/aditya/Deep_Learning_Projects/homework1/age_regression_ytr.npy")
    
    w, b = linear_regression(X_tr, ytr)


    # Report fMSE cost on the training and testing data (separately)
    fmse_train = np.sum(((np.dot(X_tr.T, w) - ytr) ** 2)/(2*X_tr.shape[1]))
    print("fMSE: ", fmse_train)

    # Testing part
    # Load data
    X_te = np.reshape(np.load("/home/aditya/Deep_Learning_Projects/homework1/age_regression_Xte.npy"), (-1, 48*48))
    X_te = np.transpose(np.hstack((X_te, np.ones((X_te.shape[0], 1)))))
    yte = np.load("/home/aditya/Deep_Learning_Projects/homework1/age_regression_yte.npy")
    w, b = linear_regression(X_tr, ytr)
    
    fmse_test = np.sum(((np.dot(X_te.T, w) - yte) ** 2)/(2*X_te.shape[1]))
    print("fMSE: ", fmse_test)

train_age_regressor()