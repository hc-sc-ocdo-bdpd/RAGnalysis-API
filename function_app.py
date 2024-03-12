import azure.functions as func
from functions.llama import llama

app = func.FunctionApp(http_auth_level=func.AuthLevel.ADMIN)

@app.route(route="llama")
def route_llama(req: func.HttpRequest) -> func.HttpResponse:
    return llama(req)
