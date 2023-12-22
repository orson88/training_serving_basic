import mlflow
from models import Model
def train_model(model, hyperparameters, data):
    experiment_name = 'amogus'
    with mlflow.start_run():
        y = [datapoint.pop(-1) for datapoint in data]
        model = Model(model, hyperparameters)
        model.fit(data, y)
        mlflow.sklearn.log_model(model, "model")

train_model(
    'Lasso Regression',
    {"alpha":0.1, "max_iter":1000, "tol": 0.01},
    [[0, 0, 0, 1], [0, 0, 0, 1], [0,1,1,1]]
)