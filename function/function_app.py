import azure.functions as func
import logging
from dotenv import load_dotenv
import os
from io import BytesIO
import json
import requests
import pandas as pd
import numpy as np
import faiss

app = func.FunctionApp(http_auth_level=func.AuthLevel.ADMIN)

@app.route(route="llama")
def route_llama(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'llama').generate()


@app.route(route="mistral")
def route_mistral(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'mistral').generate()


@app.route(route="gpt35a")
# @app.blob_input(arg_name="datablob",
#                 path="app-data/data.csv",
#                 connection="BlobStorageConnectionString")
# @app.blob_input(arg_name="indexblob",
#                 path="app-data/chunks.faiss",
#                 connection="BlobStorageConnectionString")
def route_gpt35_4k(req: func.HttpRequest) -> func.HttpResponse:
    # data = pd.read_csv(BytesIO(datablob.read()))
    # index = faiss.deserialize_index(np.frombuffer(indexblob.read(), dtype=np.uint8))

    return rag(req, 'gpt35_4k').generate()


@app.route(route="gpt35b")
def route_gpt35_16k(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'gpt35_16k').generate()


@app.route(route="gpt4")
def route_gpt4_1106(req: func.HttpRequest) -> func.HttpResponse:
    return rag(req, 'gpt4_1106').generate()


class rag():
    def __init__(self, req: func.HttpRequest, model: str):
        load_dotenv()
        self.body = req.params.get('body')

        if not self.body:
            try:
                req = req.get_json()
            except ValueError:
                pass
            else:
                self.body = req.get('body')

        self.model = model
        self.use_rag = req.params.get('use_rag') in ["True", True, None]
        self.temperature = float(req.params.get('temperature') or 0.9) 
        self.top_p = float(req.params.get('top_p') or 0.9)
        self.do_sample = req.params.get('do_sample') in ["True", True, None]
        self.frequency_penalty = float(req.params.get('frequency_penalty') or 0) 
        self.presence_penalty = float(req.params.get('presence_penalty') or 0) 
        self.max_new_tokens = int(req.params.get('max_new_tokens') or 200) 
        self.chunk_limit = int(req.params.get('chunk_limit') or 150) 
        self.k = int(req.params.get('k') or 3) 

    def generate(self) -> func.HttpResponse:
        if self.body:
            index = faiss.read_index('data/chunks.faiss')
            data = pd.read_csv('data/data.csv')
            embedding = self._embed()
            scores, ids = index.search(embedding, k=self.k)
            relevant_data = data.iloc[ids[0]]
            context = ' | '.join(relevant_data['chunks'][0:self.chunk_limit])
            self.prompt = f"Answer: {self.body} {('using: ' + context) if self.use_rag else ''}"

            if self.model in ['llama', 'mistral']:
                response = self._ml_studio_model()
            elif self.model in ['gpt35_4k', 'gpt35_16k', 'gpt4_1106']:
                response = self._ai_studio_model()

            logging.info("Response: %s", response)

            return func.HttpResponse(
                json.dumps({
                    "response": response,
                    "sources": relevant_data.assign(similarity=scores[0])[['title', 'similarity', 'url', 'chunks']].to_dict(orient='records'),
                    "parameters": self.__dict__
                }),
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                "This HTTP triggered function executed successfully. Pass a body in the query string or in the request body for a personalized response.",
                status_code=200
            )

    def _embed(self) -> list[float]:
        try:
            embedding = requests.post(
                url = "https://ragnalysis.openai.azure.com/openai/deployments/ada_embedding/embeddings?api-version=2023-05-15", 
                headers = { "Content-Type": "application/json", "api-key": os.getenv('OPENAI_KEY') }, 
                json = { "input": self.body }
            ).json()['data'][0]['embedding']
        except KeyError as e:
            raise e('ADA embedding API failed to embed: %s', self.body) 
        else:
            return np.array([embedding], dtype='float32')

    def _ml_studio_model(self) -> str:
        model = self.model.upper()
        endpoint = os.getenv(f'{model}_ENDPOINT')
        try:
            response = requests.post(
                url = f'https://ragnalysis-{endpoint}.eastus2.inference.ml.azure.com/score',
                headers = {
                    'Content-Type':'application/json',
                    'Authorization':('Bearer '+ os.getenv(f'{model}_KEY')), 
                    'azureml-model-deployment': os.getenv(f'{model}_MODEL')
                },
                json = {
                    "input_data": {
                        "input_string": [{
                            "role": "user",
                            "content": self.prompt
                        }],
                        "parameters": {
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "do_sample": self.do_sample,
                            "max_new_tokens": self.max_new_tokens
                        }
                    }
                }
            ).json()
            return response.get('output')
        except KeyError as e:
            raise e(f'ML studio model failed to generate prompt with the given context. \n \
                    Try setting use_rag to False to see if it is an issue with the context. \n \
                    Response object: {response}')

    def _ai_studio_model(self) -> str:
        try:
            response = requests.post(
                url = f"https://ragnalysis.openai.azure.com/openai/deployments/{self.model}/chat/completions?api-version=2023-05-15", 
                headers = { "Content-Type": "application/json", "api-key": os.getenv('OPENAI_KEY') }, 
                json = { 
                    "messages": [{
                        "role": "user",
                        "content": self.prompt
                    }],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    "max_tokens": self.max_new_tokens,
                    "stop": None
                }
            ).json()
            return response['choices'][0]['message'].get('content')
        except KeyError as e:
            raise e(f'AI studio model failed to generate prompt with the given context. \n \
                    Try setting use_rag to False to see if it is an issue with the context. \n \
                    Response object: {response}')
