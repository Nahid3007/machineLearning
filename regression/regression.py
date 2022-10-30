import numpy as np

def read_data(inputfile):
    data = np.loadtxt(inputfile,delimiter=',')
    X_train = data[:,:data.shape[1]-1]
    y_train = data[:,data.shape[1]-1]

    return X_train, y_train

class Normalization:
    def mean(X_train):
        mu = np.mean(X_train,axis=0)
        xmax = np.max(X_train,axis=0)
        xmin = np.min(X_train,axis=0)
    
        X_train_mean = (X_train - mu)/(xmax - xmin)

        return X_train_mean 

    def zscore(X_train):
        mu = np.mean(X_train,axis=0)
        sigma = np.std(X_train,axis=0)

        X_train_zscore = (X_train - mu)/sigma

        return X_train_zscore

class LinearRegression_:
    def __init__(self, alpha=1e-4, max_iter=10000, weights=None, bias=None, lambda_=0.):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.lambda_ = float(lambda_)
        self.weights = weights
        self.bias = bias

    def fit(self,X_train,y_train):
        num_samples = X_train.shape[0]
        try:
            num_features = X_train.shape[1]
        # if only a single feature is present
        except:
            TypeError()
            num_features = 1

        # initialize weights and bias with zeros
        if self.weights is None:
            self.weights = np.zeros(num_features)
        
        if self.bias is None:
            self.bias = np.array([0.])

        # run gradient descent
        for no_iter in range(self.max_iter):
            
            # STEP 1 : compute cost value
            cost_value = 0.
            
            for i in range(num_samples):
                y_pred = np.dot(X_train[i],self.weights) + self.bias
                sq_err = (y_pred - y_train[i])**2
                cost_value += sq_err

            cost_value = cost_value/(2*num_samples)
            
            # STEP 2 : compute gradients
            dj_dw = np.zeros(num_features)
            dj_db = 0
            
            for i in range(num_samples):
                cost = np.dot(X_train[i],self.weights) + self.bias - y_train[i]
                dj_db += cost
                for j in range(num_features):
                    try:
                        dj_dw[j] += cost * X_train[i,j]
                    # if only a single feature is present
                    except:
                        IndexError()
                        dj_dw[j] += cost * X_train[i] 

            dj_dw = dj_dw/num_samples
            dj_db = dj_db/num_samples

            # STEP 3 : update weights and bias parameters
            self.weights = self.weights - self.alpha * dj_dw
            self.bias = self.bias - self.alpha * dj_db

        return self.weights, self.bias

    def predict(self, X_train):
        y_pred = np.dot(X_train,self.weights) + self.bias

        return y_pred