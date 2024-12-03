from fastapi import APIRouter, HTTPException
from llm_integration.llm_chain import LLMIntegrationWithLLaMA
from api.schemas import GenerateRequest, GenerateResponse

# Initialize the router to manage API routes
router = APIRouter()

# Create an instance of the LLaMA integration
llm_integration = LLMIntegrationWithLLaMA()


@router.post("/generate-response", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    Endpoint to generate a response from a query and contextual documents.
    Args:
        request (GenerateRequest): The request containing a query and documents.

    Returns:
        GenerateResponse: The generated response from the model.
    """
    try:
        # Extract document texts and generate a response using LLaMA
        response = llm_integration.generate_response(
            query=request.query, retrieved_docs=[{"text": doc.text} for doc in request.documents]
        )
        return GenerateResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")


@router.get("/health")
async def health_check():
    """
    Endpoint to check the API's health status.
    Returns:
        dict: A simple dictionary indicating the API status.
    """
    return {"status": "OK"}
