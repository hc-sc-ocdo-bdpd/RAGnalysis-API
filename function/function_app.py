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


@app.route(route="gpt35a")
def route_gpt35_4k(req: func.HttpRequest) -> func.HttpResponse:
    return model(req, 'gpt35_4k')


@app.route(route="gpt35b")
def route_gpt35_16k(req: func.HttpRequest) -> func.HttpResponse:
    return model(req, 'gpt35_16k')


@app.route(route="gpt4")
def route_gpt4_1106(req: func.HttpRequest) -> func.HttpResponse:
    return model(req, 'gpt4_1106')


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
                        headers = { "Content-Type": "application/json", "api-key": os.getenv('OPENAI_KEY') }, 
                        json = { "input": body }
                    ).json()['data'][0]['embedding']

        embedding = np.array([embedding], dtype='float32')
        scores, ids = index.search(embedding, k=3)

        prompt = f"Answer: {body} {'' if not use_rag else ('using: ' + ' | '.join(data.iloc[ids[0]]['chunks'][0:150]))}"

        print("Prompt: ", prompt)

        response = ''

        if model == 'llama':
            response = ml_studio(
                prompt = prompt,
                model = model,
                endpoint = 'jludq',
                deployment = 'llama-2-7b-chat-18'
            )
        elif model == 'mistral':
            response = ml_studio(
                prompt = prompt,
                model = model,
                endpoint = 'ebbtk',
                deployment = 'mistralai-mistral-7b-instruct-5'
            )
        elif model in ['gpt35_4k', 'gpt35_16k', 'gpt4_1106']:
            response = ai_studio(
                prompt = prompt,
                model = model
            )

        return func.HttpResponse(
            json.dumps({
                "response": response,
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


def ml_studio(prompt: str, model: str, endpoint: str, deployment: str) -> str:
    response = requests.post(
        url = f'https://ragnalysis-{endpoint}.eastus2.inference.ml.azure.com/score',
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ os.getenv(f'{model.upper()}_KEY')), 'azureml-model-deployment': deployment },
        json = {
            "input_data": {
                "input_string": [{
                    "role": "user",
                    "content": prompt
                }],
                "parameters": {
                "temperature": 0.9,
                "top_p": 0.9,
                "do_sample": True,
                "max_new_tokens": 200
        }}}).json()

    return response.get('output')


def ai_studio(prompt: str, model: str) -> str:
    response = requests.post(
        url = f"https://ragnalysis.openai.azure.com/openai/deployments/{model}/chat/completions?api-version=2023-05-15", 
        headers = { "Content-Type": "application/json", "api-key": os.getenv('OPENAI_KEY') }, 
        json = { 
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 200,
            "stop": None
        }
    ).json()
    
    print("Response: ", response)
    
    return response['choices'][0]['message'].get('content')