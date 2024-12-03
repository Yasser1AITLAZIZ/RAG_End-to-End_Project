from pydantic import BaseModel
from typing import List


class Document(BaseModel):
    """
    Represents a contextual document with its text content.
    """

    text: str


class GenerateRequest(BaseModel):
    """
    Schema for the request sent to the API to generate a response.
    """

    query: str  # The user's question
    documents: List[Document]  # List of contextual documents


class GenerateResponse(BaseModel):
    """
    Schema for the response returned by the API containing the generated answer.
    """

    response: str
