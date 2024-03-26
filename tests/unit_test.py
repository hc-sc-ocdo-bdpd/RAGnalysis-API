# # https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-python?tabs=asgi%2Capplication-level&pivots=python-mode-decorators#unit-testing
# # Run w/    python3.11 -m pytest --log-cli-level=INFO tests/
import os
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'function_app'))
import logging
import unittest
import azure.functions as func

from app import route_gpt35_4k

def load_local_settings():
    with open('temp.txt', 'r') as f:
        for line in f.read().splitlines():
            key, value = line.split('=', maxsplit=1)
            os.environ[key] = value
            logging.info(key)
        logging.info(os.environ)
              

logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
load_local_settings()

class TestFunction(unittest.TestCase):

  def test_gpt35a(self):
    # Construct a mock HTTP request.
    req = func.HttpRequest(method='GET',
                           body=None,
                           url='/api/gpt35a',
                           params={
                              "body": 'What is life?',
                              "use_rag": False,
                              "temperature": 0.9,
                              "top_p": 0.9,
                              "do_sample": True,
                              "frequency_penalty": 0,
                              "presence_penalty": 0,
                              "max_new_tokens": 200,
                              "chunk_limit": 150,
                              "k": 3
                          })

    # Call the function.
    func_call = route_gpt35_4k.build().get_user_function()
    resp = func_call(req)

    logging.info(resp.get_body().decode('utf-8'))

    # Check the output.
    self.assertTrue('response' in resp.get_body().decode('utf-8'))
    self.assertTrue(resp.status_code == 200)
