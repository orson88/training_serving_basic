import mlflow
import os
import shutil
from fastapi import FastAPI, HTTPException
from typing import List
from models import Model


def get_availible_models():
    return [
        d.split('_')[1] for d in os.listdir() if os.path.isdir(d) and d.startswith("mlflow_")
        ]


app = FastAPI()


@app.get("/models")  # Получаем все варианты
def get_available_models():
    return list(Model._MODELINFO.keys())


@app.get("/models/{model}/hyperparameters")  # Получаем гиперпараметры модели
def get_model_hyperparams(model: str):
    """_summary_

    Args:
        model (str): Name of the model class you want to use. Can check 
        them from the GET /models request

    Raises:
        HTTPException: 404 If a model was not found

    Returns:
        dict: str(hyperparameter name)
    """    
    if model in Model._MODELINFO.keys():
        return Model._MODELINFO[model]['hyperparameters']
    else:
        raise HTTPException(status_code=404, detail="Model not found")


@app.post("/train/")  # Обучаем модель
def train(
        data: List[List[float]],
        hyperparameters: dict,
        model: str,
        experiment_name:str
):
    """Class to train a model

    Args:
        data (List[List[float]]): a list of lists of floats. Its the data you
        want a model to be fit on. Each list in the list is a single datapoint
        and the last element should be the target

        hyperparameters (dict): a dict of hyperparameters. You can check
        the proper dicts from GET /hyperparameters request

        model (str): the name of the model

        experiment_name (str): the name of the experiment, which you use
        to save the model and reuse it later by this field

    Raises:
        HTTPException: 404 If a model was not found
        HTTPException: 404 If the hyperparameters you passed are wrong

    Returns:
        str: done
    """
    if model not in Model._MODELINFO.keys():
        raise HTTPException(status_code=404, detail="Model not found")

    if hyperparameters.keys() != Model._MODELINFO[model]['hyperparameters'].keys():
        raise HTTPException(
            status_code=404,
            detail=f"Wrong hyperparameter structure, should be like {Model._MODELINFO[model]['hyperparameters'].keys()}"
        )

    with mlflow.start_run():
        model = Model(model, hyperparameters)
        y = [datapoint.pop(-1) for datapoint in data]
        model.train(list(data), y)
        model.save_model("testing", experiment_name)

    return 'done'


@app.get("/trained-models")
def get_trained_models():
    """A function to get all the already pretrained and saved models
    Returns:
        list: list of the names availible
    """   
    return get_availible_models()


@app.post("/predict/")
def get_prediction(name: str, data: List[List[float]]):
    """Function to get a prediction on the given data

    Args:
        name (str): name of the saved model (which was previously experiment_name)
        data (List[List[float]]): The data, same rules as in /train/

    Raises:
        HTTPException: 404 If no such pretrained model was found
        HTTPException: 404 If wrong shape of a row in data for
        the inference

    Returns:
        List: List of predictions
    """    
    if name not in get_availible_models():
        raise HTTPException(status_code=404, detail="No such model")
    model = mlflow.sklearn.load_model(f"mlflow_{name}")
    size = model.n_features_in_
    for i, datapoint in enumerate(data):
        if len(datapoint) != size:
            raise HTTPException(status_code=404, detail=f"Wrong feature count in row {i+1}, should be {size}, now is {len(datapoint)}")
    res = model.predict(data)
    return list(res)

@app.delete("/deleteModel")
def delete_model(name: str):
    """_summary_

    Args:
        name (str): _description_
    """
    if name not in get_availible_models():
        raise HTTPException(status_code=404, detail="No such model")
    shutil.rmtree(f"mlflow_{name}")

