

#LinUCB contextual bandit algorithm
**is used here to create an adaptive learning environment
as learners improve**

import numpy as np
import logging
import pickle

# Configure logging
logging.basicConfig(filename='linucb.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def save_context(context, filename="context.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(context, f)

def load_context(filename="context.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


# List of subjects
contest_rank = [
    'contest 1', 'contest 2', 'contest 3', 'contest 4', 'contest 5', 'contest 6', 'contest 7', 'contest 8', 'contest 9', 'contest 10',
    'contest 11', 'contest 12', 'contest 13', 'contest 14', 'contest 15', 'contest 16', 'contest 17', 'contest 18', 'contest 19', 'contest 20',
    'contest 21', 'contest 22', 'contest 23', 'contest 24', 'contest 25', 'contest 26', 'contest 27', 'contest 28', 'contest 29', 'contest 30',
    'contest 31', 'contest 32', 'contest 33', 'contest 34', 'contest 35', 'contest 36', 'contest 37', 'contest 38', 'contest 39', 'contest 40'
]


class LinUCB:
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha

        # Initialize parameters
        self.A = np.array([np.identity(n_features) for _ in range(n_actions)])  # action covariance matrix
        self.b = np.array([np.zeros(n_features) for _ in range(n_actions)])  # action reward vector
        self.theta = np.array([np.zeros(n_features) for _ in range(n_actions)])  # action parameter vector

        # Initialize interaction counts
        self.interaction_counts = np.zeros(n_actions)

    def predict(self, context):
        context = np.array(context)  # Convert list to ndarray
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            self.theta[a] = np.dot(np.linalg.inv(self.A[a]), self.b[a])  # theta_a = A_a^-1 * b_a
            p[a] = np.dot(self.theta[a], context) + self.alpha * np.sqrt(np.dot(context, np.dot(np.linalg.inv(self.A[a]), context)))
        return p

    def update(self, action, context, reward):
        context = np.array(context)  # Convert list to ndarray if necessary
        context = context.reshape(-1)  # Ensure context is a flat array
        self.A[action] += np.outer(context, context)  # A_a = A_a + x_t * x_t^T
        self.b[action] += reward * context  # b_a = b_a + r_t * x_t
        self.interaction_counts[action] += 1  # Increment interaction count for the chosen action

        # Log the update
        logging.info(f"Action: {action}, Context: {context.tolist()}, Reward: {reward}")
        self.save_state()

    def save_state(self, filename="model_state.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename="model_state.pkl"):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None


# Example usage

# Suppose we have 40 subjects (actions) and each context is a 4-dimensional feature vector
n_actions = len(subjects)
n_features = 1
alpha = 1.0

# Try to load an existing model, otherwise initialize a new one
modelll = LinUCB.load_state() or LinUCB(n_actions, n_features, alpha)

# Example context vector for a user (e.g., user's preferences or history in some feature space)
context = [1]


# Predict the preference scores for each subject
preference_scores = modelll.predict(context)
print("Preference scores:", preference_scores)

# Select the action (subject) with the highest score
chosen_action = np.argmax(preference_scores)
print("Chosen subject based on preference:", contest_rank[chosen_action])

# print(type(subjects.index(subjects[chosen_action])))
# Update the model with the chosen action, context, and reward (e.g., user clicked on the subject)
answer = "correct"
if answer == "incorrect":
    reward = 1
if answer == "correct":
    reward = 0
modelll.update(chosen_action, context, reward)

This section utilizes Faiss for
#Semantic search,
leveraging the "all-mpnet-base-v2" model as the sentence transformer.

!pip install -q datasets sentence-transformers faiss-cpu accelerate

from datasets import load_dataset, DatasetDict, Dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os


dataset = load_dataset('csv', data_files='/content/combined_sheets.csv')

ST = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")



sheet_name_mapping = {name: idx for idx, name in enumerate(set(dataset['train']['Year_Round']))}

# Function to embed the data and include criterion information
def embed_with_criterion(batch):

    information = [q if not p else p + " " + q for p, q in zip(batch["Preamble Text"], batch["Question"])]   # Adjust the column names if necessary
    embeddings = ST.encode(information)

    # criterion information embeddings using the mapping
    criterion_yr_rank = np.array([sheet_name_mapping[name] for name in batch["Year_Round"]], dtype=np.float32).reshape(-1, 1)
    modified_embeddings = np.hstack((embeddings, criterion_yr_rank))

    return {"embeddings": modified_embeddings}

# Apply the embedding function to the dataset
dataset = dataset.map(embed_with_criterion, batched=True, batch_size=16)

# Save the dataset and FAISS index locally
save_path = '/content/embedded_dataset'
os.makedirs(save_path, exist_ok=True)
dataset.save_to_disk(save_path)
dataset["train"].add_faiss_index(column="embeddings")
dataset["train"].save_faiss_index("embeddings", save_path + '/faiss_index')

dataset = DatasetDict.load_from_disk(save_path)
dataset["train"].load_faiss_index("embeddings", save_path + '/faiss_index')

# Function to search for the most similar entries considering the criterion
def search_with_criterion(query: str, k: int = 4, yr_rank_value= None):
    """A function that embeds a new query and returns the most probable results considering the criterion"""
    embedded_query = ST.encode(query)  # Embed new query


    criterion_yr_rank = np.array([sheet_name_mapping[yr_rank_value]], dtype=np.float32).reshape(1, 1)
    modified_query_embedding = np.hstack((embedded_query.reshape(1, -1), criterion_yr_rank))

    # Retrieve results
    scores, retrieved_examples = dataset["train"].get_nearest_examples(
        "embeddings", modified_query_embedding, k=k
    )

    return scores, retrieved_examples


year_value= "2021"
rank_value= "contest 10"
# Example usage
query = "Ask a physics question from 2021"
scores, retrieved_examples = search_with_criterion(query, k=10, yr_rank_value= f"{year_value} NSMQ {rank_value}")
print(scores)
print(retrieved_examples["Subject"])

# LLM access


!pip install -q accelerate bitsandbytes
!pip install -q oauth2client pypdf sentence_transformers
!pip install -q transformers einops accelerate bitsandbytes

!pip install -q huggingface_hub
from huggingface_hub import notebook_login
notebook_login()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# use quantization to lower GPU usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)


Round_number= [
    {
        "Round 1": "Fundamentals on Biology, Physics, Chemistry and Maths where each team receives 2/3 sets of questions depending on the stage of the competition you're in.",
        "Round 2": "Speed Race. Quick successive questioning to user and you have to answer a question as quickly and with no delay in providing answers.",
        "Round 3": "Problem of the Day. A question is posed to all three schools and given 3 mins to provide and answer to it.",
        "Round 4": "True/False. Each subject has 2 sets of questions to be answered",
        "Round 5": "Riddles. Each subject has a riddle to answer."
     }
]

storeConvo= [
    
]

def slash(query):

  templ_prompt= f"""

    From the query provided
    {query}

    extract relevant data from it
      Your output should always be in the provided JSON fomart
        -fill the most apporpriate field below
        -if apporpriate data doesn't exist in the new query put "N/A" at the space

           {{
                "year": "Put the YEAR in the query if it was provided eg. "2020" ",
                "suject": "Put the SUBJECT in the query here eg. "Chemistry" ",
                "round": "space for specific Round Number:{Round_number} eg. Round 1",
                "keywords": "Put all other keywords here. eg. what is matter? keyword is matter"
            }}
    """
  return templ_prompt
  


ASSISTANT = """
I am an assistant for high school students in Ghana.
And my goal is to help them preparing effectively for the National Science and Math Quize(NSMQ).
"""
# If you don't know the answer, just say "I do not know." Don't make up an answer.

def General_prompt(query):

    templ_prompt = f"""

    This is an important query from the user
    {query}


    You have access to all the past conversation with the user: 
    {storeConvo}

    Use the past conversations {storeConvo} and query to {query} help the user

    """

    return templ_prompt


def general_llm(prompt):
  prompt = prompt
  # Define a prompt
  # prompt = "what is ur purpose"
  # generate(formatted_prompt(prompt))

  messages = [{"role":"user","content":"hello"},
                {"role":"assistant","content":"I am an assistant"},
                {"role":"user","content":prompt}
              ]
    # tell the model to generate
  input_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
  ).to(model.device)
  outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.eos_token_id,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  response = outputs[0][input_ids.shape[-1]:]
  tokenizer.decode(response, skip_special_tokens=True)

  return(tokenizer.decode(response, skip_special_tokens=True))

import json

def verify_Q(prompt):
  

  #checks to see if the user wants to see an NSMQ question
  if prompt.startswith("/"):
    slash_prompt= slash(prompt)
    # The output of general_llm should be a JSON string
    slash_response = general_llm(slash_prompt) 
    # Attempt to decode the JSON response from general_llm
    slash_ans= json.loads(slash_response)  
    queryy= slash_ans["keywords"]
    # year_value= "2021"
    # rank_value= "contest 10"
    scores, retrieved_examples = search_with_criterion(queryy, k=5, yr_rank_value= f"{year_value} NSMQ {rank_value}")
  
    # return scores, retrieved_examples,general_llm(prompt)
    # print(scores)
    # print(retrieved_examples)
    # print(gen_llm)
    highest_value = np.argmax(scores)
    
    store= f"{retrieved_examples['Question'][highest_value]}"

    storeConvo.append({
        "user_query": prompt,
        "assistant_response": store
      })
    if len(storeConvo) > 2:
       storeConvo.pop(0)

    return str(store)

  # continue the normal conversation if not
  else:
    prompttt= General_prompt(prompt)
    store= general_llm(prompttt)
    storeConvo.append({
         "user_query": prompt,
         "assistant_response": store
        })
    if len(storeConvo) > 2:
       storeConvo.pop(0)
    return store



def access_Q(prompt):
  response= verify_Q(prompt)

  return response

access_Q("/hey can u ask me a math question")

access_Q("can u explain the question further")

print(storeConvo)

storeConvo.pop(0)

# API access point


# Install the required packages
!pip -q install fastapi uvicorn pyngrok

!ngrok authtoken 2gtKp9Zgztrv5dtK9SGsMl0cad7_5ZsXKxvA8cP6zvv4WZT94


from fastapi import FastAPI, HTTPException
from pyngrok import ngrok
import uvicorn
from threading import Thread

# Define the FastAPI app
app = FastAPI()

@app.get("/gen_llm/{prompt}")
def read_root(prompt):
    result = access_Q(prompt)
    return result

# Function to run the FastAPI app with Uvicorn
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start the FastAPI app in a new thread
server_thread = Thread(target=run_app)
server_thread.start()

!killall ngrok
# Expose the FastAPI app with ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

import requests
# Making a request to test the setup
base_url = public_url.public_url
prompt = "ask me a physics question"
response = requests.get(f"{base_url}/gen_llm/{prompt}")
print(response)
print(response.content)