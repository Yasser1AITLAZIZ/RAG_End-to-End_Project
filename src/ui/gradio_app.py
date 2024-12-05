import gradio as gr
from src.data_ingestion import DataIngestionPipeline
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_database.vector_manager import VectorManager
from src.api.api_client import APIClient
from src.retriever.retriever import Retriever
from typing import List, Tuple


class ChatbotInterface:
    """
    A class to manage the workflow of document ingestion, embedding generation,
    vector database storage, and chatbot interaction via Gradio.
    """

    def __init__(self):
        """
        Initialize all required modules for the interface.
        """
        self.pipeline = DataIngestionPipeline()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_manager = VectorManager()
        self.retriever = Retriever()
        self.api_client = APIClient()

    def process_and_store_documents(self, files: List) -> str:
        """
        Process uploaded documents, generate embeddings, and store them in the vector database.

        Args:
            files (List): List of uploaded files.

        Returns:
            str: Message indicating the result of the operation.
        """
        try:
            # Step 1: Process document
            document_text = self.pipeline.ingest(file)
            
            output_dir = "data/processed"
            os.makedirs(output_dir, exist_ok=True)

            for idx, doc in enumerate(processed_documents):
                output_path = os.path.join(output_dir, f"document_{idx+1}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=4)

            # Step 2: Generate embeddings
            embedding = self.embedding_generator.generate_embedding(document_text)

            # Step 3: Store in Pinecone
            self.vector_manager.upsert_vectors([(file.name, embedding)])
            
            return "Documents successfully processed and stored in the vector database."
        except Exception as e:
            return f"An error occurred: {e}"

    def chat_with_bot(self, query: str) -> Tuple[str, str]:
        """
        Fetch relevant documents from the vector database and interact with the chatbot.

        Args:
            query (str): User's query.

        Returns:
            Tuple[str, str]: Retrieved context and chatbot response.
        """
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query)

            # Step 2: Generate response via API
            response = self.api_client.get_response(query, retrieved_docs)
            
            # Format context for display
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
            uploaded_files = gr.File(label="Upload Documents", file_types=[".txt", ".pdf", ".docx", ".html"], file_count="multiple")
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
        chatbot_interface.process_and_store_documents,
        inputs=[uploaded_files],
        outputs=[process_output]
    )
    chat_button.click(
        chatbot_interface.chat_with_bot,
        inputs=[query_input],
        outputs=[context_display, response_output]
    )

# Run the Gradio app
if __name__ == "__main__":
    interface.launch()
