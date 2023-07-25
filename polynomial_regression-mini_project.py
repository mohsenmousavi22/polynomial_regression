#!/usr/bin/env python
# coding: utf-8

# # Polynomial Regression

# ## Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Position_Salaries.csv')


# In[3]:


x = dataset.iloc[:, 1:-1].values


# In[4]:


y = dataset.iloc[:,-1].values


# ## Training the Linear Regression model on the whole dataset

# In[5]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# ## Training the Polynomial Regression model on the whole dataset

# In[6]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)


# ## Visualising the Linear Regression results

# In[7]:


plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# ## Visualising the Polynomial Regression results (for higher resolution and smoother curve)

# In[8]:


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Predicting a new result with Linear Regression

# In[9]:


lin_reg.predict([[6.5]])


# ## Predicting a new result with Polynomial Regression

# In[10]:


lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

