import os
def get_availible_models():
    return [
        d.split('_')[1] for d in os.listdir() if os.path.isdir(d) and d.startswith("mlflow_")
        ]
