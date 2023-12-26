import pytest
from unittest.mock import MagicMock
from minio import Minio
from minio_access import list_models, create_client

def test_list_models(mocker):
    
    mock_client = MagicMock(spec=Minio)
    
    mocker.patch('minio_access.create_client', return_value=mock_client)
    
    mock_client.list_objects.return_value = [{'object_name': 'mlflow/artifact1'}, {'object_name': 'mlflow/artifact2'}]
    
    mock_client.list_objects.side_effect = [
        [{'object_name': 'mlflow/artifact1/model1'}, {'object_name': 'mlflow/artifact1/model2'}],
        [{'object_name': 'mlflow/artifact2/model3'}, {'object_name': 'mlflow/artifact2/model4'}]
    ]
    
    result = list_models()
    
    assert result == ['mlflow__artifact1/model1', 'mlflow__artifact2/model3']
