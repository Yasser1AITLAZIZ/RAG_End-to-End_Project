import gradio as gr
import os
import shutil
import requests
from typing import List, Tuple
from vector_database.vector_manager import VectorManager
from retriever.retriever import Retriever
from api.schemas import GenerateRequest, GenerateResponse, Document


class APIClient:
    """
    A client class to interact with the API endpoints.
    """

    def __init__(self, base_url: str = "http://localhost:8000/api"):
        """
        Initialize the API client with a base URL.

        Args:
            base_url (str): The base URL of the API.
        """
        self.base_url = base_url

    def generate_response(self, query: str, documents: List[dict]) -> str:
        """
        Call the /generate-response endpoint to get an LLM-generated response.

        Args:
            query (str): The user query.
            documents (List[dict]): A list of documents in the form {"text": "..."}.

        Returns:
            str: The generated response from the LLM.
        """
        payload = GenerateRequest(query=query, documents=[Document(text=doc["text"]) for doc in documents])
        response = requests.post(f"{self.base_url}/generate-response", json=payload.model_dump())

        if response.status_code != 200:
            raise ValueError(f"API request failed with status {response.status_code}: {response.text}")

        data = response.json()
        result = GenerateResponse(**data)
        return result.response


class ChatbotInterface:
    """
    A class to manage the workflow of document ingestion, embedding generation,
    vector database storage, and chatbot interaction via Gradio.
    """

    def __init__(self):
        """
        Initialize all required modules for the interface.
        """
        self.vector_manager = VectorManager()
        self.retriever = Retriever()
        self.api_client = APIClient()

    def process_and_store_documents(self, files: List) -> str:
        """
        Process uploaded documents, generate embeddings, and store them in the vector database.
        The uploaded files are copied into the 'data/raw' directory before embedding.

        Args:
            files (List): List of uploaded files.

        Returns:
            str: Message indicating the result of the operation.
        """
        try:
            output_dir = "data/raw"
            os.makedirs(output_dir, exist_ok=True)

            # Copy uploaded files into data/raw
            for f in files:
                # f is typically a tempfile.NamedTemporaryFile instance from Gradio
                shutil.copy(f.name, output_dir)

            # Generate embeddings and store in vector DB
            self.vector_manager.embed_store_db(directory_documents=output_dir)

            return "Documents successfully processed and stored in the vector database."
        except Exception as e:
            return f"An error occurred: {e}"

    def chat_with_bot(self, query: str) -> Tuple[str, str]:
        """
        Fetch relevant documents from the vector database and interact with the chatbot.
        The retrieved documents and query are sent to the /generate-response API endpoint.

        Args:
            query (str): User's query.

        Returns:
            Tuple[str, str]: (Retrieved context, chatbot response)
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query)

            # Prepare documents for the API call
            documents_for_api = [{"text": doc["text"]} for doc in retrieved_docs]

            # Get response from the LLM through the API
            response = self.api_client.generate_response(query, documents_for_api)

            # Format retrieved context for display
            context = "\n\n".join([doc["text"] for doc in retrieved_docs])
            return context, response
        except Exception as e:
            return "", f"An error occurred: {e}"


# Initialize the chatbot interface
chatbot_interface = ChatbotInterface()

# Define Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Document-Aware Chatbot")

    # Document upload section
    with gr.Row():
        with gr.Column():
            uploaded_files = gr.File(label="Upload Documents", file_types=[".pdf"], file_count="multiple")
        with gr.Column():
            process_button = gr.Button("Process and Store Documents")
            process_output = gr.Textbox(label="Processing Status", interactive=False)

    # Chatbot interaction section
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Enter your question")
        with gr.Column():
            context_display = gr.Textbox(label="Retrieved Context", lines=10, interactive=False)
            response_output = gr.Textbox(label="Bot Response", interactive=False)

    # Buttons
    chat_button = gr.Button("Ask the Bot")

    # Define button actions
    process_button.click(
        chatbot_interface.process_and_store_documents, inputs=[uploaded_files], outputs=[process_output]
    )
    chat_button.click(chatbot_interface.chat_with_bot, inputs=[query_input], outputs=[context_display, response_output])

# Run the Gradio app
if __name__ == "__main__":
    interface.launch()
