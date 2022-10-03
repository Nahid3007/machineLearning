import numpy as np

def gradient_descent(X_train,y_train,num_iter,alpha):
    
    m = X_train.shape[0] # number of training examples
    try:
        n = X_train.shape[1] # number of features
    except:
        TypeError()
        n = 1 # if only a single feature is present

    w = np.zeros(n)
    b = 0.

    # run gardient descent
    J_hist = []
    
    for no_iter in range(num_iter):
        
        # compute cost value J(w,b)
        cost_value = 0
        for i in range(m):
            fwb = np.dot(w,X_train[i]) + b
            cost = (fwb - y_train[i])**2
            cost_value += cost
            
        cost_value = cost_value/(2*m)
        J_hist.append(cost_value)
        
        # compute gradients (partial derivaties)
        dj_dw = np.zeros(n)
        dj_db = 0
        for i in range(m):
            cost = np.dot(w,X_train[i]) + b - y_train[i]
            for j in range(n):
                try:
                    dj_dw[j] += cost*X_train[i,j]
                except:
                    IndexError()
                    dj_dw[j] += cost*X_train[i] # if only a single feature is present
            dj_db += cost
         
        dj_dw = dj_dw/m
        dj_db = dj_db/m
        
        # update weight and bias parameter w,b
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
    return w,b,J_hist

def read_data(inputfile):
    data = np.loadtxt(inputfile,delimiter=',')
    X_train = data[:,:data.shape[1]-1]
    y_train = data[:,data.shape[1]-1]

    return X_train,y_train

def mean_normalization(X_train):
    mu = np.mean(X_train,axis=0)
    xmax = np.max(X_train,axis=0)
    xmin = np.min(X_train,axis=0)
    
    X_mean = (X_train - mu)/(xmax - xmin)

    return X_mean,mu,(xmax - xmin)

def zscore_normalization(X_train):
    mu = np.mean(X_train,axis=0)
    sigma = np.std(X_train,axis=0)

    X_zscore = (X_train - mu)/sigma

    return X_zscore,mu,sigma