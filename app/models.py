import mlflow
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import List
import os

experiment_name = "demo_experiment"
try:
    mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")
except Exception as e:
    pass
mlflow.set_experiment(experiment_name)



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

def gen_model_name(model, hyperparameters):
    result = model + '_'
    result += "|".join([f"{key}-{value}" for key, value in hyperparameters.items()])
    return result

def train_model(model, hyperparameters, data):
    with mlflow.start_run():
        y = [datapoint.pop(-1) for datapoint in data]
        modelname = model
        clf = Model(model, hyperparameters)
        clf = clf.model
        clf.fit(data, y)
        mlflow.sklearn.log_model(clf, gen_model_name(modelname, hyperparameters))

def use_pretrained(modelname, data):
    run_id = modelname.split('__')[0]
    model_id = modelname.split('__')[1]
    loaded_model = mlflow.sklearn.load_model(
        f"s3://mlflow/{run_id}/artifacts/{model_id}"
        )
    return loaded_model.predict(data)

