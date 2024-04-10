import os
import requests
from typing import Optional
import pandas as pd

class RagnalysisClient:
    def __init__(self, api_key: Optional[str] = None, persist: bool = False):
        self.api_key = api_key if api_key else os.getenv('FUNCTION_KEY')
        self.models = ['llama', 'mistral', 'gpt3s', 'gpt3l', 'gpt4', 'qwen']
        self.base_url = f'https://med-app.azurewebsites.net/api'
        self.persist = persist

        if not self.api_key:
            raise ValueError("API key is not provided or set in environment variables.")

    def get_models(self):
        return self.models

    def _model(self, prompt: str, model: str, use_rag: Optional[bool] = True, 
               temperature: Optional[float] = 0.9, top_p: Optional[float] = 0.9, 
               do_sample: Optional[bool] = True, frequency_penalty: Optional[float] = 0,
               presence_penalty: Optional[float] = 0, max_new_tokens: Optional[int] = 200,
               chunk_limit: Optional[int] = 150, k: Optional[int] = 3, 
               debug: Optional[bool] = False) -> dict[str: any]:
        
        try:
            response = requests.post(
                url = f"{self.base_url}/{model}?code={self.api_key}", 
                params = {
                    "body": prompt, 
                    "use_rag": use_rag,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": do_sample,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "max_new_tokens": max_new_tokens,
                    "chunk_limit": chunk_limit,
                    "k": k
                }
            )
            if not debug:
                response = response.json()
        except Exception as e:
            raise e("ERROR: API failed to process")
        else:
            if self.persist and not debug: 
                self._persist(response)
            return response
    
    def _persist(self, response):
        '''Writes the response object to a lakehouse table'''
        from pyspark.sql import SparkSession

        df = pd.json_normalize(response, sep="_")
        df['titles'], df['chunks'], df['urls'], df['similarities'] = [str([value[key] for value in df['sources'][0]]) for key in ['title', 'chunks', 'url', 'similarity']]
        df = df.drop(['sources'], axis=1)

        spark = SparkSession.builder.getOrCreate()
        spark_df = spark.createDataFrame(df)
        spark_df.write.mode("append").format("delta").saveAsTable("responses")

    def llama(self, prompt: str, **kwargs):
        return self._model(prompt, model='llama', **kwargs)

    def mistral(self, prompt: str, **kwargs):
        return self._model(prompt, model='mistral', **kwargs)

    def qwen(self, prompt: str, **kwargs):
        return self._model(prompt, model='qwen', **kwargs)

    def gpt3s(self, prompt: str, **kwargs):
        return self._model(prompt, model='gpt35a', **kwargs)

    def gpt3l(self, prompt: str, **kwargs):
        return self._model(prompt, model='gpt35b', **kwargs)

    def gpt4(self, prompt: str, **kwargs):
        return self._model(prompt, model='gpt4', **kwargs)
