#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[37]:


class LassoRegression:
    def __init__(self, alpha=0.01, max_iter=10000, tol=10):
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
    df = pd.read_csv(r"C:\Users\Alisha DJ\Downloads\Boston (1).csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


# In[38]:


df = pd.read_csv(r"C:\Users\Alisha DJ\Downloads\Boston (1).csv")


# In[39]:


# Load your dataset
X, y = dataset(r"C:\Users\Alisha DJ\Downloads\Boston (1).csv")

# Split the dataset into training and testing sets
train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)


# In[40]:


# Hyperparameter tuning

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
max_iters = [100, 500, 1000, 5000]
tols = [0.0001, 0.001, 0.01, 0.1]
best_r2 = 0
best_params = None  
best_max_iter = None

for alpha in alphas:
    for tol in tols:
        for max_iter in max_iters:
            lasso = LassoRegression(alpha=alpha, tol=tol, max_iter= max_iter)
            lasso.fit(X_train, y_train)
            y_pred = lasso.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'alpha': alpha, 'tol': tol, 'max_iter': max_iter}
            


# Train the model with best hyperparameters
lasso = LassoRegression(alpha=best_params['alpha'], tol=best_params['tol'], max_iter=best_params['max_iter'])
lasso.fit(X_train, y_train)


# In[41]:


# Evaluate the model
y_pred = lasso.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 score:", r2)
print("Best hyperparameters:", best_params)
print("best iter value:", best_max_iter)


# In[42]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Calculate R2 scores for training and testing sets for each alpha
train_r2 = []
test_r2 = []
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    lasso = LassoRegression(alpha=alpha, tol=best_params['tol'], max_iter=best_params['max_iter'])
    lasso.fit(X_train, y_train)
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)
    train_r2.append(r2_score(y_train, y_pred_train))
    test_r2.append(r2_score(y_test, y_pred_test))

# Plot R2 scores for training and testing sets
plt.plot(alphas, train_r2, label='Training R2')
plt.plot(alphas, test_r2, label='Testing R2')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 score')
plt.title('Lasso Regression R2 Scores')
plt.legend()
plt.show()


# In[43]:


import matplotlib.pyplot as plt

# Predict the target variable for the test set using the trained model
y_pred = lasso.predict(X_test)

# Calculate the R2 score for the test set
r2 = r2_score(y_test, y_pred)

# Create a scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Goodness of Fit (R2 = {r2:.2f})')
plt.show()


# In[44]:


import matplotlib.pyplot as plt

y_pred = lasso.predict(X_test)
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[21]:


import seaborn as sns

sns.histplot(y_test, kde=True, label='Actual Values')
sns.histplot(y_pred, kde=True, label='Predicted Values')
plt.xlabel('Median Value of Owner-occupied Homes in $1000s')
plt.title('Distribution Plot')
plt.legend()
plt.show()


# In[22]:


importance = lasso.coef_
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))], df.columns[:-1], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Plot')
plt.show()


# In[27]:


import statsmodels.api as sm
import matplotlib.pyplot as plt

# create QQ plot
res = sm.ProbPlot(y_test - lasso.predict(X_test), fit=True)
fig = res.qqplot(line='45')

# set plot attributes
fig.set_size_inches(8, 6)
plt.title('QQ plot of residuals')
plt.xlabel('Theoretical quantiles')
plt.ylabel('Sample quantiles')
plt.show()


# In[ ]:





# In[34]:


import matplotlib.pyplot as plt

# assuming you have stored your predicted labels in a list or numpy array called "y_pred"
# assuming you have stored your true labels in a list or numpy array called "y_true"

# plot histogram of predicted labels
plt.hist(y_pred, bins=10, alpha=0.5, label='y_pred')
# plot histogram of true labels
plt.hist(y_test, bins=10, alpha=0.5, label='y_test')
# add legend to the plot
plt.legend(loc='upper right')
# set title and axis labels
plt.title('Histogram of Predicted and True Labels')
plt.xlabel('Value')
plt.ylabel('Frequency')
# show the plot
plt.show()


# In[ ]:





# In[ ]:




