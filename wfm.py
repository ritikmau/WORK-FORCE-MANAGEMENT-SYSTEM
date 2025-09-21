import pandas as pd
import numpy as np

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import GridSearchCV


#step1 importing data
df = pd.read_csv("C:/Users/VIVEK/OneDrive/Documents/DA EPRIME/ML/allocations.csv")


y = df.iloc[:,-1]
x1 = df.iloc[:,0:3]  #character variables
x2=df.iloc[:,3:4]  #numeric variables

# step2 cleaning data

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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=0)


#forming dict and running each model one by one

models = {
    "Random Forest": RandomForestRegressor(n_estimators=10, random_state = 0, oob_score = True),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(kernel='rbf')
    }


results = []


for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append([name, r2, rmse, mae])
    print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# 7. Find best model (highest R2)
best_model = sorted(results, key=lambda x: x[1], reverse=True)[0]
print("\nBest Model:", best_model[0], "| R2:", best_model[1])


# defining hyper parameters for each model in an dict

param_grids = {
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_leaf": [1, 2, 4]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0]
    },
    "Lasso": {
        "alpha": [0.01, 0.1, 1.0, 10.0]
    },
    "SVR": {
        "C": [0.1, 1, 10],
        "epsilon": [0.1, 0.2, 0.5],
        "kernel": ["linear", "rbf"]
    }
}

# tuning best model
best_model_name = best_model[0]

if best_model_name in param_grids:
    print(f"\nRunning Hyperparameter Tuning for {best_model_name}...")
    model_class = type(models[best_model_name])
    grid = GridSearchCV(model_class(), param_grids[best_model_name], cv=5, scoring='r2')
    grid.fit(x_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best Cross-Validated R2:", grid.best_score_)

    best_model_tuned = grid.best_estimator_
    y_pred_tuned = best_model_tuned.predict(x_test)

    print("Tuned R2:", r2_score(y_test, y_pred_tuned))
else:
    print(f"No parameter grid defined for {best_model_name}.")


    
#checking if tuning is working for each model or not
#for model_name, params in param_grids.items():
#    print(f"\n--- Tuning {model_name} ---")
#    
#    # Get the model class from the original models dict
#    model_class = type(models[model_name]) if model_name in models else None
    
#    if model_class:
#        grid = GridSearchCV(model_class(), params, cv=5, scoring='r2')
#        grid.fit(x_train, y_train)
        
#        print("Best Parameters:", grid.best_params_)
#        print("Best Cross-Validated R2:", grid.best_score_)
        
#        # Evaluate tuned model on test data
#        tuned_model = grid.best_estimator_
#        y_pred_tuned = tuned_model.predict(x_test)
#        print("Test R2:", r2_score(y_test, y_pred_tuned))
#    else:
#        print(f"No model found for {model_name}")
# you can check yourself by uncmmenting this block of code

