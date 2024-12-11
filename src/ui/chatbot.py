import shutil
import os
from typing import List, Tuple
from ui.api_client import APIClient
from ui.llm_config import LLMConfig
from vector_database.vector_manager import VectorManager
from retriever.retriever import Retriever


class ChatbotInterface:
    """
    A class to manage the workflow of document ingestion, embedding generation,
    vector database storage, and chatbot interaction via the frontend.

    Responsibilities:
    - Process and store documents in a vector database.
    - Retrieve relevant context from the vector database.
    - Interact with the backend API to generate responses.
    - Manage LLM configuration (temperature, max_tokens).
    """

    def __init__(self) -> None:
        """
        Initialize the chatbot interface by setting up the vector manager,
        retriever, API client, and default LLM configuration.
        """
        self.vector_manager = VectorManager("data/raw/")
        self.retriever = Retriever()
        self.api_client = APIClient(os.getenv("api_service_address", "http://localhost:8080/api"))
        self.llm_config = LLMConfig()

    def process_and_store_documents(self, files: List) -> str:
        """
        Process uploaded documents and store them in the vector database.

        Args:
            files (List): A list of uploaded file objects.

        Returns:
            str: A status message indicating success or error.
        """
        try:
            output_dir = "data/raw"
            os.makedirs(output_dir, exist_ok=True)

            for f in files:
                shutil.copy(f.name, output_dir)

            self.vector_manager.embed_store_db(directory_documents=output_dir)
            return "Documents successfully processed and stored in the vector database."
        except Exception as e:
            return f"An error occurred: {e}"

    def chat_with_bot(self, query: str) -> Tuple[str, str]:
        """
        Chat with the bot by retrieving relevant documents and generating a response
        using the current LLM configuration parameters.

        Args:
            query (str): The user's query.

        Returns:
            Tuple[str, str]: The retrieved context and the bot's generated response.
        """
        try:
            retrieved_docs = self.retriever.retrieve(query)
            documents_for_api = [{"text": doc["text"]} for doc in retrieved_docs]
            response = self.api_client.generate_response(
                query, documents_for_api, temperature=self.llm_config.temperature, max_tokens=self.llm_config.max_tokens
            )

            context = "\n\n".join([doc["text"] for doc in retrieved_docs])
            return context, response
        except Exception as e:
            return "", f"An error occurred: {e}"

    def clear_index_and_raw_folder(self) -> str:
        """
        Clear all vectors stored in the vector database and remove the files in /data/raw.
        Then recreate an empty index to ensure a fresh start.

        Returns:
            str: Message indicating the result of the operation.
        """
        try:
            self.vector_manager.delete_index()
            self.vector_manager.client.create_index(self.vector_manager.index_name, self.vector_manager.dimensions)

            raw_folder = "data/raw"
            if os.path.exists(raw_folder):
                shutil.rmtree(raw_folder)

            return "Cleanup completed successfully! A new empty index has been created."
        except Exception as e:
            return f"An error occurred during cleanup: {e}"

    def list_documents(self) -> List[str]:
        """
        List the documents currently stored in the raw data folder.

        Returns:
            List[str]: A list of document filenames.
        """
        raw_folder = "data/raw"
        if not os.path.exists(raw_folder):
            return []
        return os.listdir(raw_folder)

    def update_temperature(self, temp: float) -> str:
        """
        Update the LLM's temperature setting.

        Args:
            temp (float): The new temperature value.

        Returns:
            str: A confirmation message.
        """
        self.llm_config.update_temperature(temp)
        return f"Updated temperature to {temp}"

    def update_max_tokens(self, max_t: int) -> str:
        """
        Update the LLM's max tokens setting.

        Args:
            max_t (int): The new maximum tokens value.

        Returns:
            str: A confirmation message.
        """
        self.llm_config.update_max_tokens(max_t)
        return f"Updated max tokens to {max_t}"
