import azure.functions as func
import logging
from dotenv import load_dotenv
import os
import json
import requests
import pandas as pd
import numpy as np
import faiss

def llama(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    load_dotenv()

    body = req.params.get('body')
    # body = req.get('body')
    if not body:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            body = req_body.get('body')

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

        response = requests.post(
            'https://ragnalysis-ebbtk.eastus2.inference.ml.azure.com/score',
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ os.getenv('LLAMA_KEY')), 'azureml-model-deployment': 'mistralai-mistral-7b-instruct-5' },
            json = {
                "input_data": {
                    "input_string": [
                    {
                        "role": "user",
                        "content": f"{body} using the context {' | '.join(data.iloc[ids[0]]['text'])}"
                    }],
                    "parameters": {
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 200
            }}}).json()

        print("response ", response)

        return func.HttpResponse(
            json.dumps({
                "response": response.get('output'),
                "source": data.iloc[ids[0]].assign(similarity=scores[0])[['title', 'similarity']].to_string(index=False)
            }),
            mimetype="application/json"
        )

    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a body in the query string or in the request body for a personalized response.",
            status_code=200
        )
