#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn


# ## Because the previous dataset that I'm using,didn't contain much information thats suits for linear regression(Predicting a continuous value)). 
# 
# ## Today I'm using another dataset that is about the personal medical cost.
# 

# In[208]:


data=pd.read_csv('./insurance.csv')
print(data.info())
print('-----------------------------------------------------------------')
print(data.head())


# ## Firstly, we can find out there is no null value in the data set.
# 
# * age : Indicates the age of the person. It contains data of type "int64".
# * sex : It refers to the gender of the person. It contains "object" type data.
# * bmi : It refers to the Body Mass Index of the person and contains the data of type "float64". BMI is a measure of the weight of a person, divided by the square of its length. Determines the person's obesity value. The formula for USA and METRIC units is as follows:
# * children : It refers to the number of children that a person has. It contains data of type "int64".
# * smoker : Indicates whether the person smokes or not. It contains "object" type data.
# * region : Specifies which region the person is from. It contains "object" type data.
# * charges : The person's total insurance premium is specified. Although not specified, it is assumed to be in US dollars. It contains "float64" type data.

# In[160]:


data.describe()


# In[86]:


data.corr()


# # Basic understanding about dataset

# In[77]:


fig, axes = plt.subplots(figsize=(10, 10))  # This method creates a figure and a set of subplots
sns.heatmap(data=data.corr(), annot=True, linewidths=.5, ax=axes)  # Figure out heatmap
# Parameters:
# data : 2D data for the heatmap.
# annot : If True, write the data value in each cell.
# linewidths : Width of the lines that will divide each cell.
# ax : Axes in which to draw the plot, otherwise use the currently-active Axes.
plt.show()  # Shows only plot and remove other informations


# ## Data visualition

# In[78]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
data.plot(kind="hist", y="age", bins=70, color="#728CA3", ax=axes[0][0])
data.plot(kind="hist", y="bmi", bins=200, color="#73C0F4", ax=axes[0][1])
data.plot(kind="hist", y="children", bins=5, color="#E6EFF3", ax=axes[1][0])
data.plot(kind="hist", y="charges", bins=200, color="#5A4D4C", ax=axes[1][1])
plt.show()


# ### By the charts above we can know the distribution of each feature. 

# In[191]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
data.plot(kind='scatter', x='age', y='charges', alpha=0.5, color='#14325C', ax=axes[0], title="Age vs. Charges")
data.plot(kind='scatter', x='bmi', y='charges', alpha=0.5, color='#5398D9', ax=axes[1], title="Sex vs. Charges")
data.plot(kind='scatter', x='children', y='charges', alpha=0.5, color='#D96B0C', ax=axes[2], title="Children vs. Charges")
plt.show()


# ### As far as age is concerned, we can find out the medical charges are positive proportional to the age, which is uqite make sense. 
# ### As for BMI,  we do can find out those who have high BMI, is more likely to pay higher medical charges.

# In[192]:


sns.scatterplot(x="bmi", y="charges", data=data, palette='Set2', hue='smoker')


# ### We can easily found out that under the same BMI the mediacal charges of smoker is significantly higher than the non-smokers.
# ----

# ### And we convert the smoker and sex to numeric value.

# In[209]:


data.smoker=data.smoker.map({'yes':1,'no':0})
data.smoker.value_counts()


# In[210]:


data.sex=data.sex.map({'male':1,'female':0})
data.sex.value_counts()


# ## Split the dataset to X and Y

# In[211]:


data.drop(['region'],axis=1,inplace=True)

X=data.drop(['charges'],axis=1)
Y=data['charges']

print(X.head())


# In[212]:


X["bmi"] = (X.bmi - np.min(X.bmi))/(np.max(X.bmi) - np.min(X.bmi))
print(X.head())


# In[213]:


from sklearn.model_selection import train_test_split  # Import "train_test_split" method
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=38)
print(x_train)
print(y_train)


# ----
# ## Simple linear regression :
# ### we choose BMI as x,to predict the corresponding charges.

# In[214]:


simple_linear_reg=LinearRegression()
simple_linear_reg.fit(x_train.bmi.values.reshape(-1,1),y_train)
Prediction=simple_linear_reg.predict(x_test.bmi.values.reshape(-1,1))

plt.figure(figsize=(16,8))

plt.scatter(X.bmi,Y,c='black',label='Origianl')
plt.scatter(x_test.bmi,Prediction,c='blue',label='Prediction')
plt.legend()
plt.show()


# ### We can easily see that because the data is too complicated, the results predicted by simple linear regression is not very accurate.
# -----
# 
# ### So following I will use some regression model try to have a better result.

# ## Multiple linear regression

# In[215]:


multiple_linear_reg = LinearRegression(fit_intercept=False,normalize=True)  # Create a instance for Linear Regression model
multiple_linear_reg.fit(x_train, y_train)


# ## Polynomial linear regression

# In[216]:


polynomial_features = PolynomialFeatures(degree=3)  # Create a PolynomialFeatures instance in degree 3
x_train_poly = polynomial_features.fit_transform(x_train)  # Fit and transform the training data to polynomial
x_test_poly = polynomial_features.fit_transform(x_test)  # Fit and transform the testing data to polynomial

polynomial_reg = LinearRegression(fit_intercept=False)  # Create a instance for Linear Regression model
polynomial_reg.fit(x_train_poly, y_train)  # Fit data to the model


# ## Decision tree regression

# In[217]:


from sklearn.tree import DecisionTreeRegressor  # Import Decision Tree Regression model

decision_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=13)  # Create a instance for Decision Tree Regression model
decision_tree_reg.fit(x_train, y_train)  # Fit data to the model


# ## Random forest regrssion

# In[218]:


from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regression model

random_forest_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=13)  # Create a instance for Random Forest Regression model
random_forest_reg.fit(x_train, y_train)  # Fit data to the model


# ---
# ## Evaluation Model : Because it's hard to visualize the result of each model. In this section we will do some measurements to evaluate the performance on the models we fit. 
# ## I will use following metric to determine the performance of each model
# * R2 score
# * RMSE score
# 
# 

# In[219]:


from sklearn.model_selection import cross_val_predict  # For K-Fold Cross Validation
from sklearn.metrics import r2_score  # For find accuracy with R2 Score
from sklearn.metrics import mean_squared_error  # For MSE
from math import sqrt  # For squareroot operation


# **Evaluating Multiple Linear Regression Model**

# In[221]:


# Prediction with training dataset:
y_pred_MLR_train = multiple_linear_reg.predict(x_train)
# Prediction with testing dataset:
y_pred_MLR_test = multiple_linear_reg.predict(x_test)

# Find training accuracy for this model:
accuracy_MLR_train = r2_score(y_train, y_pred_MLR_train)
print("Training Accuracy for Multiple Linear Regression Model: ", accuracy_MLR_train)

# Find testing accuracy for this model:
accuracy_MLR_test = r2_score(y_test, y_pred_MLR_test)
print("Testing Accuracy for Multiple Linear Regression Model: ", accuracy_MLR_test)

# Find RMSE for training data:
RMSE_MLR_train = sqrt(mean_squared_error(y_train, y_pred_MLR_train))
print("RMSE for Training Data: ", RMSE_MLR_train)

# Find RMSE for testing data:
RMSE_MLR_test = sqrt(mean_squared_error(y_test, y_pred_MLR_test))
print("RMSE for Testing Data: ", RMSE_MLR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_MLR = cross_val_predict(multiple_linear_reg, X, Y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_MLR = r2_score(Y, y_pred_cv_MLR)
print("Accuracy for 10-Fold Cross Predicted Multiple Linaer Regression Model: ", accuracy_cv_MLR)


# **Evaluating Polynomial Regression Model**

# In[222]:


y_pred_PR_train = polynomial_reg.predict(x_train_poly)

# Prediction with testing dataset:
y_pred_PR_test = polynomial_reg.predict(x_test_poly)

# Find training accuracy for this model:
accuracy_PR_train = r2_score(y_train, y_pred_PR_train)
print("Training Accuracy for Polynomial Regression Model: ", accuracy_PR_train)

# Find testing accuracy for this model:
accuracy_PR_test = r2_score(y_test, y_pred_PR_test)
print("Testing Accuracy for Polynomial Regression Model: ", accuracy_PR_test)

# Find RMSE for training data:
RMSE_PR_train = sqrt(mean_squared_error(y_train, y_pred_PR_train))
print("RMSE for Training Data: ", RMSE_PR_train)

# Find RMSE for testing data:
RMSE_PR_test = sqrt(mean_squared_error(y_test, y_pred_PR_test))
print("RMSE for Testing Data: ", RMSE_PR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_PR = cross_val_predict(polynomial_reg, polynomial_features.fit_transform(X), Y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_PR = r2_score(Y, y_pred_cv_PR)
print("Accuracy for 10-Fold Cross Predicted Polynomial Regression Model: ", accuracy_cv_PR)


# **Evaluating Decision Tree Regression Mode**

# In[223]:


# Prediction with training dataset:
y_pred_DTR_train = decision_tree_reg.predict(x_train)

# Prediction with testing dataset:
y_pred_DTR_test = decision_tree_reg.predict(x_test)

# Find training accuracy for this model:
accuracy_DTR_train = r2_score(y_train, y_pred_DTR_train)
print("Training Accuracy for Decision Tree Regression Model: ", accuracy_DTR_train)

# Find testing accuracy for this model:
accuracy_DTR_test = r2_score(y_test, y_pred_DTR_test)
print("Testing Accuracy for Decision Tree Regression Model: ", accuracy_DTR_test)

# Find RMSE for training data:
RMSE_DTR_train = sqrt(mean_squared_error(y_train, y_pred_DTR_train))
print("RMSE for Training Data: ", RMSE_DTR_train)

# Find RMSE for testing data:
RMSE_DTR_test = sqrt(mean_squared_error(y_test, y_pred_DTR_test))
print("RMSE for Testing Data: ", RMSE_DTR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_DTR = cross_val_predict(decision_tree_reg, X, Y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_DTR = r2_score(Y, y_pred_cv_DTR)
print("Accuracy for 10-Fold Cross Predicted Decision Tree Regression Model: ", accuracy_cv_DTR)


# **Evaluate for random forest**

# In[224]:


# Prediction with training dataset:
y_pred_RFR_train = random_forest_reg.predict(x_train)

# Prediction with testing dataset:
y_pred_RFR_test = random_forest_reg.predict(x_test)

# Find training accuracy for this model:
accuracy_RFR_train = r2_score(y_train, y_pred_RFR_train)
print("Training Accuracy for Random Forest Regression Model: ", accuracy_RFR_train)

# Find testing accuracy for this model:
accuracy_RFR_test = r2_score(y_test, y_pred_RFR_test)
print("Testing Accuracy for Random Forest Regression Model: ", accuracy_RFR_test)

# Find RMSE for training data:
RMSE_RFR_train = sqrt(mean_squared_error(y_train, y_pred_RFR_train))
print("RMSE for Training Data: ", RMSE_RFR_train)

# Find RMSE for testing data:
RMSE_RFR_test = sqrt(mean_squared_error(y_test, y_pred_RFR_test))
print("RMSE for Testing Data: ", RMSE_RFR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_RFR = cross_val_predict(random_forest_reg, X, Y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_RFR = r2_score(Y, y_pred_cv_RFR)
print("Accuracy for 10-Fold Cross Predicted Random Forest Regression Model: ", accuracy_cv_RFR)


# In[225]:


training_accuracies = [accuracy_MLR_train, accuracy_PR_train, accuracy_DTR_train, accuracy_RFR_train]
testing_accuracies = [accuracy_MLR_test, accuracy_PR_test, accuracy_DTR_test, accuracy_RFR_test]
training_RMSE = [RMSE_MLR_train, RMSE_PR_train, RMSE_DTR_train, RMSE_RFR_train]
testing_RMSE = [RMSE_MLR_test, RMSE_PR_test, RMSE_DTR_test, RMSE_RFR_test]
cv_accuracies = [accuracy_cv_MLR, accuracy_cv_PR, accuracy_cv_DTR, accuracy_cv_RFR]
parameters = ["fit_intercept=False", "fit_intercept=False", "max_depth=5", "n_estimators=400, max_depth=5"]
table_data = {"Parameters": parameters, "Training Accuracy": training_accuracies, "Testing Accuracy": testing_accuracies, 
              "Training RMSE": training_RMSE, "Testing RMSE": testing_RMSE, "10-Fold Score": cv_accuracies}
model_names = ["Multiple Linear Regression", "Polynomial Regression", "Decision Tree Regression", "Random Forest Regression"]

table_dataframe = pd.DataFrame(data=table_data, index=model_names)
table_dataframe


# **Now let's compare the training and testing accuracy of each model:**

# In[226]:


table_dataframe.iloc[:, 1:3].plot(kind="bar", ylim=[0.0, 1.0])


# **Let's compare each model's training and testing RMSE:**

# In[228]:


table_dataframe.iloc[:, 3:5].plot(kind="bar", ylim=[0.0, 8000])


# **Finally, compare the score values for 10-Fold Cross Validation:**
# 

# In[229]:


table_dataframe.iloc[:, 5].plot(kind="bar", ylim=[0.0, 1.0])


# **As you can see, The result predicted by the multiple linear regreession is the least accurate, which is roughly 70% accuracy.
# As for polynomail , decision tree , random forest regression, all of them are approximately 80% ~ 85% accuracy.**
# 
# 
# **And the random forest regression has the highest accuracy, which is roughly 85%.**
# 
