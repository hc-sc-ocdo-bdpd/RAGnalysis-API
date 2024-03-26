import os
from time import time
import tempfile
import tiktoken
from azure.storage.blob import BlobClient


def read_blob(blob_name: str, operation):
    blob = BlobClient(
                account_url=os.environ['STORAGE_URL'], 
                container_name=os.environ['STORAGE_CONTAINER'], 
                blob_name=blob_name, 
                credential=os.environ['STORAGE_KEY']
            )
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(blob.download_blob().readall())
        temp_file.flush()
        return operation(temp_file.name)


def timer(some_function):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        t2 = time()
        return result, t2 - t1
    return wrapper


def count_tokens(text: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    ntokens = len(encoding.encode(text))
    return ntokens