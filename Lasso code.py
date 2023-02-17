#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math


# In[ ]:


class Lasso_Regression():
    def __init__(self, int, reg_factor, n_iterations=5000, learning_rate=0.01, lambda_parameter):
    self.int = int
    self.regularization = l1_regularization(alpha=reg_factor)
    super(LassoRegression, self).__init__(n_iteration, learning_rate)


# In[ ]:


def fit(self, X, Y):
    self.m, self = X.shape
    self.b = 0
    self.w = np.zeros(self.n)
    self.X = X
    self.Y = Y
    super(LassoRegression, self).fit(X, y)
    for i in range(self.iterations) :
        self.update_weights()
        return self
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




