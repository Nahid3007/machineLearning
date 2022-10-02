import numpy as np

def gradient_descent(X_train,y_train,w_init,b_init,num_iter,alpha):
    
    m = X_train.shape[0] # number of training examples
    try:
        n = X_train.shape[1] # number of features
    except:
        TypeError()
        n = 1 # if only a single feature is present

    w = w_init
    b = b_init

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
            if n == 1:
                dj_db += cost[0]
            else:
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
    X_train_mean = np.empty_like(X_train)
    X_train_mu = np.empty_like(X_train)
    for i in range(X_train.shape[1]):
        mu = np.mean(X_train[:,i])
        X_train_mu[:,i] = X_train[:,i] - mu
        xmax = np.max(X_train[:,i])
        xmin = np.min(X_train[:,i])
    
        X_train_mean[:,i] = X_train_mu[:,i]/(xmax - xmin)

    return X_train_mean,X_train_mu

def zscore_normalization(X_train):
    X_train_zscore = np.empty_like(X_train)
    X_train_mu = np.empty_like(X_train)
    for i in range(X_train.shape[1]):
        mu = np.mean(X_train[:,i])
        sigma = np.std(X_train[:,i])
        X_train_mu[:,i] = X_train[:,i] - mu
        
        X_train_zscore[:,i] = X_train_mu[:,i]/sigma

    return X_train_zscore,X_train_mu