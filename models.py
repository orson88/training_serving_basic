import mlflow
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import List

class Model:
    _MODELINFO = {
    "Linear Regression": {
        "model": LinearRegression(),
        "hyperparameters": {}
    },
    "Ridge Regression": {
        "model": Ridge(),
        "hyperparameters": {
            "alpha": [0.1, 1.0, 10.0],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        }
    },
    "Lasso Regression": {
        "model": Lasso(),
        "hyperparameters": {
            "alpha": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000, 3000],
            "tol": [0.001, 0.01, 0.1]
        }
    },
    "Decision Tree Regressor": {
        "model": DecisionTreeRegressor(),
        "hyperparameters": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "Random Forest Regressor": {
        "model": RandomForestRegressor(),
        "hyperparameters": {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    }
    }
    
    def __init__(self, model_name: str, hyperparameters: dict):
        self.model = self.__class__._MODELINFO[model_name]['model']
        self.hyperparameters = hyperparameters
        self.trained = False
    
    def train(self, data: List[List[float]], y):
        self.model.fit(data, y)
        self.trained = True

    def save_model(self, model_name, experiment_name):
        if self.trained:
            mlflow.sklearn.save_model(self.model, f"mlflow_{experiment_name}")
        else:
            print('model not trained')

