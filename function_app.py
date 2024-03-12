import azure.functions as func
import logging
from dotenv import load_dotenv
import os
import json
import requests
import pandas as pd
import numpy as np
import faiss

app = func.FunctionApp(http_auth_level=func.AuthLevel.ADMIN)

@app.route(route="llama")
def route_llama(req: func.HttpRequest) -> func.HttpResponse:
    return model(req, 'llama')


@app.route(route="mistral")
def route_mistral(req: func.HttpRequest) -> func.HttpResponse:
    return model(req, 'mistral')


def model(req: func.HttpRequest, model: str) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    load_dotenv()
    body = req.params.get('body')
    use_rag = req.params.get('use_rag')

    if not body:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            body = req_body.get('body')
            use_rag = req_body.get('use_rag')

    use_rag = True if use_rag in ['1', 'True', 'true', 1] else False

    if body:
        index = faiss.read_index('data/chunks.faiss')
        data = pd.read_csv('data/data.csv')

        embedding = requests.post(
                        url = "https://ragnalysis.openai.azure.com/openai/deployments/ada_embedding/embeddings?api-version=2023-05-15", 
                        headers = { "Content-Type": "application/json", "api-key": os.getenv('EMBEDDING_KEY') }, 
                        json = { "input": body }
                    ).json()['data'][0]['embedding']

        embedding = np.array([embedding], dtype='float32')
        scores, ids = index.search(embedding, k=3)

        print(data.iloc[ids[0]].assign(similarity=scores[0])[['title', 'similarity']].to_string())

        endpoint = ''
        deployment = ''

        if model == 'llama':
            endpoint = 'jludq'
            deployment = 'llama-2-7b-chat-18'
        elif model == 'mistral':
            endpoint = 'ebbtk'
            deployment = 'mistralai-mistral-7b-instruct-5'

        response = requests.post(
            f'https://ragnalysis-{endpoint}.eastus2.inference.ml.azure.com/score',
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ os.getenv(f'{model.upper()}_KEY')), 'azureml-model-deployment': deployment },
            json = {
                "input_data": {
                    "input_string": [{
                        "role": "user",
                        "content": f"{body} {'' if not use_rag else ('using the context' + ' | '.join(data.iloc[ids[0]]['text'][0:100]))}"
                    }],
                    "parameters": {
                    "temperature": 0.9,
                    "top_p": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 200
            }}}).json()

        # print("response ", response)
        print("RAG: ", bool(use_rag), use_rag)
        # print("prompt ",  f"{body} {'' if not bool(req.params.get('use_rag')) else ('using the context' + ' | '.join(data.iloc[ids[0]]['text'][0:100]))}")

        return func.HttpResponse(
            json.dumps({
                "response": response.get('output'),
                "source": data.iloc[ids[0]].assign(similarity=scores[0])[['title', 'similarity']].to_string(index=False),
                "context": list(data.iloc[ids[0]]['chunks'])
            }),
            mimetype="application/json"
        )

    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a body in the query string or in the request body for a personalized response.",
            status_code=200
        )
