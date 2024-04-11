import os
from time import time
import tempfile
import tiktoken
from azure.storage.blob import BlobClient


def read_blob(blob_name: str, operation):
    """Loads from Azure blob storage to a local variable
    
    Args:
        blob_name : Name of the file on blob storage
        operation: The function to perform on the blob (ex pd.read_csv, faiss.read_index)

    """
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


def timer(some_function) -> float:
    """Decorator for timing function runtime in seconds"""

    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        t2 = time()
        return result, t2 - t1
    return wrapper


def count_tokens(text: str, model_name: str) -> int:
    """Count the number of tokens in a string
    
    Args:
        model_name: The model used to count tokens (ex. gpt-3.5-turbo, text-embedding-ada-002)
    
    """
    encoding = tiktoken.encoding_for_model(model_name)
    ntokens = len(encoding.encode(text))
    return ntokens