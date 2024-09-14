from pymongo import MongoClient
import requests
from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pydantic import BaseModel

# MongoDB Atlas setup
client = MongoClient("mongodb+srv://swapnilsingh:ganpati@cluster0.ydftg.mongodb.net/")
db = client['vt_chatbot']
pages_collection = db['pages']

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index for storing embeddings
embedding_dim = 384  # For the 'all-MiniLM-L6-v2' model
index = faiss.IndexFlatL2(embedding_dim)

# Retrieve pages from MongoDB
def get_pages_from_mongo():
    return list(pages_collection.find({}, {'_id': 1, 'content': 1, 'title': 1, 'url': 1}))

# Create embeddings and store them in FAISS
def create_embeddings_and_store():
    pages = get_pages_from_mongo()
    for i, page in enumerate(pages):
        content_embedding = model.encode(page['content'])
        index.add(np.array([content_embedding]))  # Add embedding to FAISS
        # Update MongoDB with FAISS index position (i.e., store i in MongoDB)
        pages_collection.update_one({'_id': page['_id']}, {'$set': {'faiss_index': i}})
        print(f"Processed {page['title']}")

# Create embeddings for all scraped pages (Uncomment to run this only once)
# create_embeddings_and_store()

# Pydantic model for request validation
class QueryRequest(BaseModel):
    query: str  # The query field must be a string

app = FastAPI()

# Load the generative model (GPT)
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to perform query and retrieve relevant documents using FAISS
def get_relevant_docs(query, k=10):
    query_embedding = model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    print(f"Query Embedding: {query_embedding}")
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    relevant_docs = []
    for i in indices[0]:
        doc = pages_collection.find_one({'faiss_index': int(i)})
        if doc:
            print(f"Found Document: {doc['title']}")
            relevant_docs.append(doc['content'])
        else:
            print(f"No document found for index: {i}")
    return relevant_docs

# Function to generate a response using GPT-2
def generate_response(retrieved_docs, query):
    context = "\n".join(retrieved_docs)
    if not context:
        context = "No relevant documents found."
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    input_ids = gpt_tokenizer.encode(input_text, return_tensors='pt')
    output = gpt_model.generate(input_ids, max_length=200, num_return_sequences=1)
    response = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@app.post("/query")
async def query_vt_chatbot(request: QueryRequest):
    # Step 1: Retrieve relevant documents
    relevant_docs = get_relevant_docs(request.query)
    
    # Step 2: Generate response
    response = generate_response(relevant_docs, request.query)
    return {"response": response}