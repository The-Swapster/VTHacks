from pymongo import MongoClient
from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaTokenizer, LlamaForCausalLM
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import json
import torch

# MongoDB Atlas setup
client = MongoClient("mongodb+srv://swapnilsingh:ganpati@cluster0.ydftg.mongodb.net/")
db = client['vt_chatbot']
pages_collection = db['pages']

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pydantic model for request validation
class QueryRequest(BaseModel):
    query: str  # The query field must be a string

app = FastAPI()

# # Load the generative model (GPT)
# gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
llama_model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3')
llama_tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-3')

# Function to perform query and retrieve relevant documents using Atlas Vector Search
def get_relevant_docs(query, k=3):
    # Convert query to embedding
    query_embedding = model.encode(query).tolist()  # Convert to list
    #print("Query Embedding:", json.dumps(query_embedding))  # Debug print

    # Perform aggregation query with $search
    response = pages_collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": 4,
                "index":"PlotSemanticSearch"
        }
        }
    ])
    
    # Convert response to list
    relevant_docs = [doc['content'] for doc in response]
    print(relevant_docs)
    return relevant_docs

# def generate_response(retrieved_docs, query):
#     context = "\n".join(retrieved_docs)[:1024]  # Limit the context size for GPT-2
#     input_text = f"Query: {query}\nContext: {context}\nAnswer:"
#     input_ids = llama_tokenizer.encode(input_text, return_tensors='pt')

#     # Create an attention mask for the input_ids
#     # Ensure to handle case where pad_token_id might be None
#     pad_token_id = llama_tokenizer.pad_token_id
#     if pad_token_id is not None:
#         attention_mask = (input_ids != pad_token_id).long()  # Convert boolean mask to long tensor
#     else:
#         attention_mask = torch.ones_like(input_ids)  # If no pad_token_id, just use ones

#     # Generate response
#     output = llama_model.generate(
#         input_ids,
#         attention_mask=attention_mask,  # Pass the attention mask
#         max_length=500
#     )
#     return llama_tokenizer.decode(output[0], skip_special_tokens=True)

# @app.post("/query")
# async def query_vt_chatbot(request: QueryRequest):
#     # Step 1: Retrieve relevant documents
#     relevant_docs = get_relevant_docs(request.query)
    
#     # Step 2: Generate response
#     response = generate_response(relevant_docs, request.query)
#     return {"response": response}

def generate_response(retrieved_docs, query):
    context = "\n".join(retrieved_docs)[:1024]  # Limit the context size
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    input_ids = llama_tokenizer.encode(input_text, return_tensors='pt')

    # Create an attention mask for the input_ids
    pad_token_id = llama_tokenizer.pad_token_id
    if pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()  # Convert boolean mask to long tensor
    else:
        attention_mask = torch.ones_like(input_ids)  # If no pad_token_id, just use ones

    # Generate response
    output = llama_model.generate(
        input_ids,
        attention_mask=attention_mask,  # Pass the attention mask
        max_length=500,
        num_return_sequences=1,  # Return only one sequence
        pad_token_id=pad_token_id  # Ensure padding token id is used
    )
    return llama_tokenizer.decode(output[0], skip_special_tokens=True)

@app.post("/query")
async def query_vt_chatbot(request: QueryRequest):
    # Step 1: Retrieve relevant documents
    relevant_docs = get_relevant_docs(request.query)
    
    # Step 2: Generate response
    response = generate_response(relevant_docs, request.query)
    return {"response": response}