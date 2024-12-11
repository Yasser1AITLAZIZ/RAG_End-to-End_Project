from api.schemas import GenerateRequest, GenerateResponse, Document
import requests
from typing import List


class APIClient:
    """
    A client class to interact with the backend API endpoints.
    """

    def __init__(self, base_url: str):
        """
        Initialize the API client with a base URL.

        Args:
            base_url (str): The base URL for the API endpoints.
        """
        self.base_url = base_url

    def generate_response(self, query: str, documents: List[dict], temperature: float, max_tokens: int) -> str:
        """
        Send a request to the backend to generate a response based on a query and retrieved documents,
        specifying the temperature and max_tokens for the language model.

        Args:
            query (str): The user's query.
            documents (List[dict]): A list of documents (text segments) retrieved from the vector database.
            temperature (float): The creativity parameter for the LM's responses.
            max_tokens (int): The maximum number of tokens for the LM's response.

        Returns:
            str: The generated response from the backend.
        """
        payload = GenerateRequest(
            query=query,
            documents=[Document(text=doc["text"]) for doc in documents],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Add temperature and max_tokens to the request payload
        payload_data = payload.model_dump()

        response = requests.post(f"{self.base_url}/generate-response", json=payload_data)

        if response.status_code != 200:
            raise ValueError(f"API request failed with status {response.status_code}: {response.text}")

        data = response.json()
        result = GenerateResponse(**data)
        return result.response
