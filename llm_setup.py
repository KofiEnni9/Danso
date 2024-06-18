import warnings
warnings.filterwarnings('ignore')

import locale
locale.getpreferredencoding = lambda: "UTF-8"

!pip install -q gspread
!pip install -q oauth2client pypdf sentence_transformers
!pip install -q huggingface_hub llama_index llama-index-llms-huggingface llama-index-readers-file
!pip install -q transformers einops accelerate langchain bitsandbytes llama-index-embeddings-langchain langchain-community


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


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids(["<|eot_id|>", "<|im_end|>"])
]

Round_number= [
    {
        "Round 1"= "Fundamentals on Biology, Physics, Chemistry and Maths where each team receives 2/3 sets of questions depending on the stage of the competition you're in.",
        "Round 2"= "Speed Race. Quick successive questioning to user and you have to answer a question as quickly and with no delay in providing answers.", 
        "Round 3"= "Problem of the Day. A question is posed to all three schools and given 3 mins to provide and answer to it.", 
        "Round 4"= "True/False. Each subject has 2 sets of questions to be answered", 
        "Round 5"= "Riddles. Each subject has a riddle to answer."
     }
]

ASSISTANT = """
I am an assistant for high school students in Ghana.
And my goal is to help them preparing effectively for the National Science and Math Quize(NSMQ).
"""
# If you don't know the answer, just say "I do not know." Don't make up an answer.

def General_prompt(query):

    templ_prompt = f"""system

    From the query provided
    {query}

    extract relevant data from it


    Your output should always be in the provided JSON fomart
        -fill the most apporpriate field below
        -if apporpriate data doesn't exist in the new query put "N/A" at the space

           {{
                "quiz": {{
                    "isAsk": "leave this space as "True" if the user wants to BE ASKED question(s)/quized/tested ",
                    "question": "this space is for the exact question the user wants TO BE ASKED"
                }},
                "year": "Put the YEAR the query here",
                "suject": "Put the SUBJECT in the query here",
                "round": "space specific Round Number:{Round_number}"
                "general_Q": "Put the answer to question the user ASKED here"
            }}
    End immidetly afer this
    <|im_end|>
    Edge cases you must handle:
    - If the user request has completely nothing to do with NSMQ, you will respond politely that you cannot help.<|im_end|>
    """

   # Call the LLM to get the completion
    return templ_prompt



# Define a prompt
prompt = "Can you ask me a question from 2021 in physics"
# generate(formatted_prompt(prompt))

# def generate(formatted_prompt):
# formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
messages = [{"role":"user","content":"hello"},
              {"role":"assistant","content":ASSISTANT},
              {"role":"user","content":formatted_prompt(prompt)}
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
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
tokenizer.decode(response, skip_special_tokens=True)




########################################################################################
#######################################################################################
!pip install -q datasets sentence-transformers faiss-cpu accelerate

from datasets import load_dataset, DatasetDict, Dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os


dataset = load_dataset('csv', data_files='/content/combined_sheets.csv')

ST = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Create a mapping from SheetName to integers
sheet_name_mapping = {name: idx for idx, name in enumerate(set(dataset['train']['SheetName']))}

# Function to embed the data and include criterion information
def embed_with_criterion(batch):

    information = [q if not p else p + " " + q for p, q in zip(batch["Preamble Text"], batch["Question"])]   # Adjust the column names if necessary
    embeddings = ST.encode(information)

    # Add criterion information to embeddings using the mapping
    criterion_info = np.array([sheet_name_mapping[name] for name in batch["SheetName"]], dtype=np.float32).reshape(-1, 1)
    modified_embeddings = np.hstack((embeddings, criterion_info))

    return {"embeddings": modified_embeddings}

# Apply the embedding function to the dataset
dataset = dataset.map(embed_with_criterion, batched=True, batch_size=16)

# Save the dataset and FAISS index locally
save_path = '/content/n_embedded_dataset'
os.makedirs(save_path, exist_ok=True)
dataset.save_to_disk(save_path)
dataset["train"].add_faiss_index(column="embeddings")
dataset["train"].save_faiss_index("embeddings", save_path + '/faiss_index')

dataset = DatasetDict.load_from_disk(save_path)
dataset["train"].load_faiss_index("embeddings", save_path + '/faiss_index')

# Function to search for the most similar entries considering the criterion
def search_with_criterion(query: str, k: int = 4, criterion_value="Round 1"):
    """A function that embeds a new query and returns the most probable results considering the criterion"""
    embedded_query = ST.encode(query)  # Embed new query

    
    criterion_info = np.array([sheet_name_mapping[criterion_value]], dtype=np.float32).reshape(1, 1)
    modified_query_embedding = np.hstack((embedded_query.reshape(1, -1), criterion_info))
    
    # Retrieve results
    scores, retrieved_examples = dataset["train"].get_nearest_examples(
        "embeddings", modified_query_embedding, k=k
    )
    
    return scores, retrieved_examples

# Example usage
query = "experiment to determine the acceleration due to gravity"
scores, retrieved_examples = search_with_criterion(query, k=5, round_value= round_value, subject_value= subject_value, rank_value= round_value)
print(scores)
print(retrieved_examples["Question"])















def FULL_Quize(query):

    explain_prompt = f"""system

        Your main purpose is to make sure the user understands CONCEPTS
        
        These are your past conversations 

        Your output should always be in the provided JSON fomart
            -fill the most apporpriate field
            -if apporpriate data doesn't exist in the new query put "N/A" at the space

            {{
                    "explain": "put the answer to the users questions here. Break your explainations down to help the user undersatnd well"
                }}
        End immidetly afer this
        <|im_end|>
        Edge cases you must handle:
        - If the user request has completely nothing to do with NSMQ, you will respond politely that you cannot help.<|im_end|>
        """


    ready_prompt = f"""system

    Your have three main purpose:
     1.Find out if the user is ready for the QUESTIONS TO BE ASKED
     2.Explain any thing they don't understand about the quiz 

     
    You will be asking a student certain questions 
    In which the student would have different time durations to answer the questions

    "Round 1"-> "questions time frame 30s", "Fundamentals on Biology, Physics", Chemistry and Maths where each team receives 2/3 sets of questions depending on the stage of the competition you're in.",
    "Round 2"-> "questions time frame must be answered as soon as possible", "Speed Race. Quick successive questioning to user and you have to answer a question as quickly and with no delay in providing answers.", 
    "Round 3"-> "questions time frame 30s", "Problem of the Day. A question is posed to all three schools and given 3 mins to provide and answer to it.", 
    "Round 4"-> "questions time frame 30s", "True/False. Each subject has 2 sets of questions to be answered", 
    "Round 5"-> "questions time frame 60s", "Riddles. Each subject has a riddle to answer."

    I have given you all the information to answer user's prompts
    Your goal is to find out if the user is ready to ba quized 

    
    Your output should always be in the provided JSON fomart
        -fill the most apporpriate field
        -if apporpriate data doesn't exist in the new query put "N/A" at the space

           {{
                "isReady": "leave this space as "True" if the user confirms to be ready for the questions"
                "pre-test": "put the answer to the users questions here"
            }}
    End immidetly afer this
    <|im_end|>
    Edge cases you must handle:
    - If the user request has completely nothing to do with NSMQ, you will respond politely that you cannot help.<|im_end|>
    """

   # Call the LLM to get the completion
    return templ_prompt

