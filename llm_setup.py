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

system_prompt = """
I am an assistant for high school students in Ghana.
And my goal is to help them preparing effectively for the National Science and Math Quize(NSMQ).
"""
# If you don't know the answer, just say "I do not know." Don't make up an answer.

def formatted_prompt(query):

    templ_prompt = f"""system

    From the query provided
    {query}

    extract relevant data from it


    Your output should always be in the provided JSON fomart
        -fill the most apporpriate field below
        -if apporpriate data doesn't exist in the new query put "N/A" at the space

           {{
                "quiz": "leave this space as "True" if the user wants to be asked question(s)/quized/tested ",
                "year": "this space is for the question outputed",
                "suject": "this space is for the answer to the question asked",
                "round": "space for Round Number:{Round_number}"
                "everythingelse": "anything that does not fall under year, round, subject goes here"
            }}
    End immidetly afer this
    <|im_end|>
    Edge cases you must handle:
    - If the user request has completely nothing to do with NSMQ, you will respond politely that you cannot help.<|im_end|>
    """

   # Call the LLM to get the completion
    return templ_prompt


self_pace= """
    

"""


# Define a prompt
prompt = "Can you ask me a question from 2021 in physics"
# generate(formatted_prompt(prompt))

# def generate(formatted_prompt):
# formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
messages = [{"role":"user","content":"hello"},
              {"role":"assistant","content":system_prompt},
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



###########################################################################
###########################################################################


!pip install -q datasets sentence-transformers faiss-cpu accelerate

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


dataset = load_dataset('csv', data_files='/content/combined_sheets.csv')

dataset

ST = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed(batch):
    information = batch["Question"]  # Adjust the column name if necessary
    return {"embeddings" : ST.encode(information)}


dataset = dataset.map(embed,batched=True,batch_size=16)

dataset.push_to_hub("Ennin/d_sets", token="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", revision="embedded")

from datasets import load_dataset

dataset = load_dataset("Ennin/d_sets",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings")

def search(query: str, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples

scores , result = search("", 4 ) 

result["sheetName"]

##############################################################################################
##############################################################################################

RAG chatbot



pip install -q datasets sentence-transformers faiss-cpu accelerate bitsandbytes

from sentence_transformers import SentenceTransformer
from datasets import load_dataset

ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("not-lain/wikipedia",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings") # column name that has the embeddings of the dataset

def search(query: str, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples
