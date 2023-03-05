#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[29]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=0.0001):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.mean(y)
        
        for iteration in range(self.max_iter):
            for j in range(n_features):
                X_j = X[:, j]
                y_pred = self.predict(X)
                r = y - y_pred + self.coef_[j] * X_j
                z = np.dot(X_j, X_j)
                self.coef_[j] = self._soft_thresholding_operator(np.dot(X_j, r) / z, self.alpha / z)
                
            y_pred = self.predict(X)
            residual = y - y_pred
            RSS = np.dot(residual, residual)
            if iteration > 0 and abs(RSS - RSS_old) < self.tol:
                break
            RSS_old = RSS
            
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    def _soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0
    
def dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


# In[30]:


# Load your dataset
X, y = dataset(r"C:\Users\Alisha DJ\Downloads\Boston (1).csv")

# Split the dataset into training and testing sets
train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)


# In[31]:


# Hyperparameter tuning
alphas = np.logspace(-3, 3, 7) # different alpha values 
tolerances = np.logspace(-5, -3, 3) #different tolerance values 
best_r2 = 0
best_params = None

for alpha in alphas:
    for tol in tolerances:
        lasso = LassoRegression(alpha=alpha, tol=tol)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_params = {'alpha': alpha, 'tol': tol}

# Train the model with best hyperparameters
lasso = LassoRegression(alpha=best_params['alpha'], tol=best_params['tol'])
lasso.fit(X_train, y_train)


# In[32]:


# Evaluate the model
y_pred = lasso.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 score:", r2)
print("Best hyperparameters:", best_params)


# In[ ]:





# In[ ]:




