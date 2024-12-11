from typing import List
from llm_integration.config import MAX_TOKENS, TEMPERATURE, LLM_API_KEY, DEFAULT_MODEL
from llm_integration.exceptions import LLMChainError
from groq import Groq


class LLMIntegrationWithLLaMA:
    """
    Integration with LLaMA 3 using Groq API.
    """

    def __init__(self):
        """
        Initialize the LLM integration with the Groq API and LLaMA model parameters.
        """
        try:
            self.client = Groq(api_key=LLM_API_KEY)  # Initialize the Groq client
            self.model = DEFAULT_MODEL
            # self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        except Exception as e:
            raise LLMChainError(f"Failed to initialize Groq client: {e}")

    def generate_response(self, query: str, retrieved_docs: List[dict], temperature: float, max_tokens: int) -> str:
        """
        Generate a response from the LLM using Groq API.

        Args:
            query (str): The user's query.
            retrieved_docs (List[dict]): List of retrieved documents.
            temperature (float): creativity of model.
            max_tokens (int)

        Returns:
            str: The response generated by the LLM.
        """
        try:
            # Format the context
            context = "\n".join([f"Document {i + 1}:\n{doc['text']}" for i, doc in enumerate(retrieved_docs)])

            # Construct the full input prompt
            prompt = (
                "You are a knowledgeable assistant. Use the following documents to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

            # Generate response using the Groq API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                stream=False,
            )

            # Extract the assistant's reply
            response_content = completion.choices[0].message.content.strip()
            return response_content
        except Exception as e:
            raise LLMChainError(f"Failed to generate response with Groq LLaMA: {e}")
