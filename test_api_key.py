from dotenv import load_dotenv
import requests
import json
import os

keys = load_dotenv()
api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
query = '"Digital Rocks Portal"'

url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
query_params = {
    "query": query,
    "fields": "title,url,publicationTypes,publicationDate,openAccessPdf",
    "year": "2023-",
}

headers = {"x-api-key": api_key}

response = requests.get(url, params=query_params, headers=headers).json()
print(json.dumps(response, indent=2))
