import pytest
from src.llm_integration.llm_chain import LLMIntegrationWithLLaMA


@pytest.fixture(scope="module")
def setup_llm_chain():
    """
    Fixture to initialize the LLMIntegrationWithLLaMA instance.
    """
    return LLMIntegrationWithLLaMA()


# def test_count_tokens(setup_llm_chain):
#    """
# Test the count_tokens method to ensure it returns the correct number of tokens.
# """
# llm = setup_llm_chain
# sample_text = "This is a test sentence."
# token_count = llm.count_tokens(sample_text)

# assert isinstance(token_count, int), "Token count should be an integer."
# assert token_count > 0, "Token count should be greater than zero."
# assert token_count < MAX_TOKENS, f"Token count should be less than {MAX_TOKENS}."


def test_generate_response_real_api(setup_llm_chain):
    """
    Test the generate_response method using the real LLaMA API via Groq.
    """
    llm = setup_llm_chain

    # Test query
    query = "What are the benefits of the OptimRiskMaximizer method?"
    retrieved_docs = [
        {"text": "The OptimRiskMaximizer is a method for optimizing risk management in trading."},
        {"text": "It balances risk and reward dynamically based on market conditions."},
    ]

    # Call the real API
    response = llm.generate_response(query, retrieved_docs)

    assert isinstance(response, str), "Response should be a string."
    assert len(response) > 0, "Response should not be empty."
    assert "OptimRiskMaximizer" in response, "Response should mention 'OptimRiskMaximizer'."
