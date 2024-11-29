from utils.helpers import get_api_key
from vector_database.exceptions import APIKeyError


try:
    # Retrieve the API key
    api_key = get_api_key()

except FileNotFoundError as fnf_error:
    # Handle the case where the .env file is missing
    print(f"Error: {fnf_error}")

except APIKeyError as api_error:
    # Handle the case where the API key is missing in the .env file
    print(f"Error: {api_error}")

except Exception as e:
    # Handle any other unforeseen exceptions
    print(f"An unexpected error occurred: {e}")


PINECONE_API_KEY = api_key
PINECONE_ENVIRONMENT = "us-east-1"
DEFAULT_INDEX_NAME = "vector-index-t"
DEFAULT_DIMENSIONS = 3