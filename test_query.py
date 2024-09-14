import requests

# Define the URL of the FastAPI server (adjust if not running locally)
url = "http://127.0.0.1:8000/query"

# Define the query you want to test
query = {"query": "What are the available courses at VT?"}

# Send a POST request
response = requests.post(url, json=query)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    response_data = response.json()
    print("Response from Chatbot:")
    print(f"Query: {response_data.get('query')}")
    print(f"Retrieved Documents: {response_data.get('retrieved_docs')}")
    print(f"Generated Answer: {response_data.get('generated_answer')}")
else:
    print(f"Failed to get a response, status code: {response.status_code}")