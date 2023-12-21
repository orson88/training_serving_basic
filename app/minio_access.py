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
