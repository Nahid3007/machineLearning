# Linear Regression

Assume we have the following `linear model` to make the prediction:

$$ f_{w,b}(x^{(i)}) = w\cdot x^{(i)} + b \tag{1} \ , $$

where $x^{(i)}$ is the feature value for the $i$-th training example and $w,b$ are the weight and bias parameters. For linear regession the there is only one single feature as an input.

The `cost function` $J(w,b)$ is a measurement which describes how good the `predicted values` $f_{w,b}(x^{(i)})$ matches to the `target values` $y^{(i)}$ of the training set. In general to get a good fit of predicted to target values the cost function needs be minimzed.

$$ J(w,b) = \frac{1}{2m} \sum_i^m (f_{w,b}(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} \sum_i^m (w\cdot x^{(i)} + b - y^{(i)})^2 \tag{2} \ .$$

In order to get the "optimal" values for $w$ and $b$ the `gradient descent method` will be applied to find a *`local`* or *`global minima`* of the cost function.

$$w = w - \alpha \cdot \frac{\partial}{\partial w} J(w,b) \tag{3} \ ,$$

$$b = b - \alpha \cdot \frac{\partial}{\partial w} J(w,b) \tag{4} \ ,$$

where $\alpha$ is the `learning rate value`. Chosing a very small value of $\alpha$ will lead to the local or global minima, but it will take large number of iteration to converge. A large value of $\alpha$ may lead to an overestimated value for the cost function $J$. After a certain iteration the soltion might diverge and the local or global minima cannot be found.

Evaluating the the `partial derivatives` of the cost function will give the following for $w$ and $b$:

$$w = w - \alpha \cdot \frac{1}{m} \sum_i^m (w\cdot x^{(i)} + b - y^{(i)}) x^{(i)} \tag{5} \ , $$

$$b = b - \alpha \cdot \frac{1}{m} \sum_i^m (w\cdot x^{(i)} + b - y^{(i)}) \tag{6} \ . $$

# Multiple Linear Regression

For a multiple linear regression the number of feature $x_j^{(i)}$ is at least greater than 1.

$$ \textbf{x}^{(i)} = \left( x_1^{(i)}, x_2^{(i)}, \ldots, x_n^{(i)} \right) \tag{1} \ ,$$

where $n$ is the total number of feature and $i= 1 \ldots m$ is the number of training examples.

Assume here we have only one training example $(i=1)$ and the number of total features is $n$ the linear model is

$$ f_{\textbf{w},b}(\textbf{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + \ldots + w_n x_n  + b \tag{2} \ . $$

The general `linear model` to make the prediction looks as follow:

$$ f_{\textbf{w},b}(\textbf{x}^{(i)}) = \textbf{w} \cdot \textbf{x}^{(i)} + b \tag{3} \ , $$

where $\textbf{w}$ , $\mathbf{x}^{(i)}$ are vectors now and the symbol $\cdot$ is the `vector dot product`.

The `cost function` $J$ for the multiple linear regression yields to

$$ J(\textbf{w},b) = \frac{1}{2m} \sum_i^m (f_{\textbf{w},b}(\textbf{x}^{(i)}) - y^{(i)})^2 = \frac{1}{2m} \sum_i^m (\textbf{w} \cdot \textbf{x}^{(i)} + b - y^{(i)})^2 \tag{4} \ .$$

For the gradient descent the `partial derivates` for the parameter $\mathbf{w}, b$ need be calculated.

$$ \frac{\partial}{\partial \mathbf{w}_j} J(\textbf{w},b) = \frac{1}{m} \sum_i^m (\textbf{w} \cdot \textbf{x}^{(i)} + b - y^{(i)}) x_j^{(i)} \tag{5} \ ,$$

$$ \frac{\partial}{\partial b} J(\textbf{w},b) = \frac{1}{m} \sum_i^m (\textbf{w} \cdot \textbf{x}^{(i)} + b - y^{(i)}) \tag{6} \ .$$

Evaluating the the partial derivatives of the cost function will give the following for $\mathbf{w}$ and $b$

$$\mathbf{w}_j = \mathbf{w}_j - \alpha \cdot \frac{1}{m} \sum_i^m (\textbf{w} \cdot \textbf{x}^{(i)} + b - y^{(i)}) x_j^{(i)} \tag{7} \ , $$

$$b = b - \alpha \cdot \frac{1}{m} \sum_i^m (\textbf{w} \cdot \textbf{x}^{(i)} + b - y^{(i)}) \tag{9} \ . $$

# Feature Scaling

## Mean Normalization

$$ \mathbf{x}_j = \frac{\mathbf{x}_j - \mu_j}{x_{j,max} - x_{j,min}} \tag{1} \ ,$$

where $\mu_n$ is the mean value of the feature vector $\mathbf{x}_j$ over all training examples.

## Z-Score Normalization

$$ \mathbf{x}_j = \frac{\mathbf{x}_j - \mu_j}{\sigma_j} \tag{2} \ ,$$

where $\mu_n$ is the mean value and $\sigma_j$ is the standard deviation of the feature vector $\mathbf{x}_j$ over all training examples.