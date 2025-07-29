import pandas as pd
import numpy as np

#step1 importing data
df = pd.read_csv("C:/Users/VIVEK/OneDrive/Documents/DA EPRIME/ML/allocations.csv")


y = df.iloc[:,-1]
x1 = df.iloc[:,0:3]  #character variables
x2=df.iloc[:,3:4]  #numeric variables

# step2 cleaning data
from sklearn.preprocessing import LabelEncoder

#converting character to numeric
x1 = x1.apply(LabelEncoder().fit_transform)

# to see what is in LabelEncoder()
LabelEncoder_x = LabelEncoder()


#since all the values are now in numeric format
x = pd.concat((x1,x2), axis = 1)

#since we have done converting x1 to numeric and then concatinating it to x2 
# we will delete it

del x1,x2

#step 3 model training
# ML model
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state = 0, oob_score = True)
#n_estimator is the number of decision trees to be made it is determined by optimal tree concept
#random_state tells model to start with this sample for first whenever it is executed  to get same outputs otherwise trees will be built different everytime
#oob_score 
# to see all the parameters and their explanation select the randomforestregressor and press ctrl+i

#used to train the model by syntax: model.fit(x,y) with parameters
regressor.fit(x, y)

#trying changing nestmiters 
regressor = RandomForestRegressor(n_estimators=200, random_state = 0, oob_score = True)
regressor.fit(x, y)


#evaluating model using gridsearch for best value of tree and leaf size and deapth. You can pass all the possible values as a parameters
# using hyperperameters to improve accuracy from overfitting
from sklearn.model_selection import GridSearchCV
#creates a grid and pick x values sequencially

#now leaving the regressor behind, creating new object for the same by using hyper parameter
params= {"n_estimators": [5,10,25,50,200,150,250,300], "min_samples_leaf":[1,2,3], "max_depth": [i for i in range(1,11)]}
#here we have stored the parameters in an object

model = GridSearchCV(regressor, params, cv=3)
rf_result = model.fit(x,y)

rf_result.best_estimator_
regressor = RandomForestRegressor(n_estimators=25, random_state = 0, oob_score = True)
regressor.fit(x, y)


