import azure.functions as func
from backend import rag

app = func.FunctionApp(http_auth_level=func.AuthLevel.ADMIN)

@app.route(route="llama")
def route_llama(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'llama').generate()


@app.route(route="mistral")
def route_mistral(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'mistral').generate()


@app.route(route="qwen")
def route_qwen(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'qwen').generate()


@app.route(route="gpt35a")
# @app.blob_input(arg_name="datablob", path="app-data/data.csv", connection="BlobStorageConnectionString")
# @app.blob_input(arg_name="indexblob", path="app-data/chunks.faiss", connection="BlobStorageConnectionString")
def route_gpt35_4k(req: func.HttpRequest) -> func.HttpResponse:
    # data = pd.read_csv(BytesIO(datablob.read()))
    return rag(req, 'gpt35_4k').generate()


@app.route(route="gpt35b")
def route_gpt35_16k(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'gpt35_16k').generate()


@app.route(route="gpt4")
def route_gpt4_1106(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'gpt4_1106').generate()


