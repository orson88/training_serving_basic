from minio import Minio

def create_client():
    client = Minio("localhost:9110",
        access_key="abobusamogus",
        secret_key="darkmagapatriot",
        secure=False
    )
    return client

def save_model(model_filename: str):
    client = create_client()
    client.fput_object(
            "models", model_filename, model_filename,
        )
    print(
        model_filename.split('/')[-1], "successfully uploaded as object",
        model_filename.split('/')[-1], "to bucket", "models"
        )

def list_models():
    client = create_client()
    objects = client.list_objects("mlflow")
    outs = []
    for run in objects:
        path = [_.object_name for _ in client.list_objects("mlflow", run.object_name+"artifacts/")]
        modelname = path[0].split('/')[-2]
        outs.append(path[0].split('/')[0]+"__"+modelname)
    return outs

def delete_model(modelname: str):
    client = create_client()
    run_id = modelname.split('__')[0]
    path1 = [_.object_name for _ in client.list_objects("mlflow", run_id)]
    path2 = [_.object_name for _ in client.list_objects("mlflow", path1)]
    paths3 = [_.object_name for _ in client.list_objects("mlflow", path2)]
    paths4 = [_.object_name for _ in client.list_objects("mlflow", paths3)]
    for path in paths4:
        client.remove_object("mlflow", path)
    return "done"


