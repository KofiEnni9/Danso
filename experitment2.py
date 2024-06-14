import warnings
warnings.filterwarnings('ignore')

import locale
locale.getpreferredencoding = lambda: "UTF-8"

from google.colab import drive
drive.mount('/content/drive')

!pip install -q gspread oauth2client

!pip install -q pypdf sentence_transformers huggingface_hub llama_index llama-index-llms-huggingface llama-index-readers-file
!pip install -q transformers einops accelerate langchain bitsandbytes
!pip install -q llama-index-embeddings-langchain
!pip install -q langchain-community

from oauth2client.service_account import ServiceAccountCredentials
import gspread
from googleapiclient.discovery import build
from google.colab import auth

# Authenticate and create the PyDrive client
auth.authenticate_user()

# Path to service account key file
SERVICE_ACCOUNT_FILE = '/content/drive/My Drive/secrets/sublime-command-414712.json'

# Define the scopes
SCOPES = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# Authenticate using a service account
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPES)
gc = gspread.authorize(credentials)
drive_service = build('drive', 'v3', credentials=credentials)

from googleapiclient.discovery import build
import gspread
import io
import csv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.readers.file import CSVReader

from google.colab import auth
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd

# Authenticate and create the PyDrive client
auth.authenticate_user()

# Path to service account key file
SERVICE_ACCOUNT_FILE = '/content/drive/My Drive/secrets/sublime-command-414712.json'

# Define the scopes
SCOPES = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# Authenticate using a service account
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPES)
gc = gspread.authorize(credentials)
drive_service = build('drive', 'v3', credentials=credentials)


# Define your folder ID
folder_id = '1q1Lyjgs-chG-HUIMxaXIJBPv-AwnZ1Ug'

# List all files in the specified folder
query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'"
results = drive_service.files().list(q=query).execute()
items = results.get('files', [])

if not items:
    print('No files found in the specified folder.')
else:
    print(f"Found {len(items)} files in the specified folder.")


all_sheets_df = []

for item in items:
    file_id = item['id']
    file_name = item['name']
    try:
        sheet = gc.open_by_key(file_id)
        
        for worksheet in sheet.worksheets():
            worksheet_title = worksheet.title
            data = worksheet.get_all_values()
            
            # Convert data to a DataFrame
            df = pd.DataFrame(data)
            
            # Optionally set the first row as headers if necessary
            df.columns = df.iloc[0]
            df = df[1:]
            
            df['SheetName'] = worksheet_title
            df['FileName'] = file_name
            
            all_sheets_df.append(df)
    except Exception as e:
        print(f"Failed to read file {file_name} with id {file_id}: {e}")

# Combine all dataframes into one
combined_df = pd.concat(all_sheets_df, ignore_index=True)

# Save the combined dataframe to a CSV file
csv_filename = '/content/combined_sheets.csv'
combined_df.to_csv(csv_filename, index=False)

print(f"Combined CSV file created at: {csv_filename}")

parser = CSVReader()
file_extractor = {".csv": parser}  # Add other CSV formats as needed
documents = SimpleDirectoryReader(
    "/content/", file_extractor=file_extractor
).load_data()

import gc
import json
import re
import xml.etree.ElementTree as ET
from functools import partial
from typing import get_type_hints

import transformers
import torch

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate

from huggingface_hub import notebook_login
notebook_login()

system_prompt = """
You are an expert when it comes to the National Science and Math Quize(NSMQ) held in Ghana for high school students.
Your goal is to help students prepare effectively for NSMQ. 
"""
#format supported by LLama2
query_wrapper_prompt = PromptTemplate(
    "<|USER|>{query_str}<|ASSISTANT|>"
)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",

    model_kwargs={"torch_dtype": torch.float16 , "load_in_4bit":True} )

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.indices.service_context import ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)


service_context

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()

!pip install -q langchain-community

!pip install -q langchain

from langchain.chains.openai_functions import convert_to_openai_function
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.pydantic_v1 import BaseModel, Field, validator

# class Question(BaseModel):
#     """Asks the user a question that is categorized under a certain Round and year"""
#     round_year: str = Field(description="year and round category that a qestion should fall under")
#     question: str = Field(description="answer is the question-to-be-asked")

#     @validator("round_year")
#     def round_and_year_must_not_be_empty(cls, field):
#         if not field:
#             raise ValueError("round_and_year_cannot be empty.")
#         return field

def retrieve_question(year: str) -> str:
  year = f"the year to retrive questions from is {year}"
  return year


# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "retrieve_question",
#             "description": "request to be quizzed/asked questions/tested",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "transaction_id": {
#                         "type": "string",
#                         "description": "The quiz year",
#                     }
#                 },
#                 "required": ["year"],
#             },
#         },
#     },
#   ]


# Function to generate response from the model
def generate_func(query):
   
    prompt = f"""system

    From the query provided 
        {query}

    extract relevant data from it


    Your output should always be in the provided JSON fomart
        -fill the most apporpriate field below
        -if data does't  with "N/A" if such data is not availble

           {{
                "year": "this space is for the question outputed",
                "suject": "this space is for the answer to the question asked",
                "topic": "this space is for the explanation to the question you asked",
                "round": "",
                "everythingelse": "anything that does not fall under question, answer, explanation goes here"
            }}


    Edge cases you must handle:
    - If there are no functions that match the user request, you will respond politely that you cannot help.<|im_end|>
    
    <|im_start|>assistant"""


    completion = query_engine.query(prompt).response
    return completion

prompt = " Ask me a question from 2021"
completion = generate_func(prompt)
print(f"{completion}")


  # if functions:
  #     print(functions)
  # else:
  #     print(completion.strip())
  # print("="*100)

prompt = " what is the capital of the upperwest of ghana"
completion = generate_func(prompt)
print(f"{completion}")