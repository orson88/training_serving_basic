import os
from typing import List
import shutil
import mlflow
from fastapi import FastAPI, HTTPException
from models import Model, train_model, use_pretrained
from minio_access import list_models, delete_model

EXPERIMENT_NAME = "demo_experiment"
try:
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location="s3://mlflow")
except Exception as e:
    pass
mlflow.set_experiment(EXPERIMENT_NAME)


app = FastAPI()

4
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
        model: str
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
            detail=f"Wrong hyperparameter structure"
        )
    train_model(model, hyperparameters, data)
    return 'done'


@app.get("/trained-models")
def get_trained_models():
    """A function to get all the already pretrained and saved models
    Returns:
        list: list of the names availible
    """   
    return list_models()


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
    if name not in list_models():
        raise HTTPException(status_code=404, detail="No such model")

    res = use_pretrained(name, data)
    return list(res)

@app.delete("/deleteModel")
def deleteModel(name: str):
    """_summary_

    Args:
        name (str): _description_
    """
    if name not in list_models():
        raise HTTPException(status_code=404, detail="No such model")
        
    return delete_model(name)


