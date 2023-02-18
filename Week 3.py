#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the library
import numpy as np


# In[ ]:


#now let's create a class for Lasso Regression
class LassoRegression:
    def __init__(self, alpha=1, max_iter=1000, lamda_parameter):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lamda_parameter = lamda_parameter
        self.intercept_ = None
        self.coef_ = None


# In[ ]:


#Now lets fit the dataset to the model
   def fit(self, X, y):
       
       # n_samples --> no of datapoints
       # n_features --> no of input features
       n_samples, n_features = X.shape
       self.intercept_ = 0
       self.coef_ = np.zeros(n_features)
       


# In[ ]:


# function for updating the weight & bias value
   def upadte_weights(self):

   # linear equation of the model
   Y_prediction = self.predict(self.X)

   # gradients (dw, db)

   # gradient for weight
   dw = np.zeros(self.n_features)

   for i in range(self.n_features):

     if self.w[i]>0:

       dw[i] = (-(2*(self.X[:,i]).dot(self.Y - Y_prediction)) + self.lambda_parameter) / self.n_samples
   
     else :

       dw[i] = (-(2*(self.X[:,i]).dot(self.Y - Y_prediction)) - self.lambda_parameter) / self.n_samples


# In[ ]:


# Now let's write gradient for bias
  db = - 2 * np.sum(self.Y - Y_prediction) / self.n_samples


  # For updating the weights & bias
  self.w = self.w - self.learning_rate*dw
  self.b = self.b - self.learning_rate*db


# In[ ]:



  # To Predict the Target variable
    def predict(self,X):

        return X.dot(self.w) + self.b

