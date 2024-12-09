import gradio as gr
from ui.chatbot import ChatbotInterface


class Frontend:
    """
    A class to handle the frontend layout and logic using Gradio.
    Provides a more sophisticated UI with multiple sections, tabs, and improved user experience.
    """

    def __init__(self, chatbot_interface: ChatbotInterface) -> None:
        """
        Initialize the frontend with references to the chatbot interface for backend operations.

        Args:
            chatbot_interface (ChatbotInterface): An instance of ChatbotInterface for backend interactions.
        """
        self.chatbot_interface = chatbot_interface
        self.interface = None

    def build_interface(self) -> None:
        """
        Build the Gradio interface with a more structured Advanced Settings tab.

        Changes:
        - Group the LLM parameters (temperature, max tokens) under a labeled box for clarity.
        - Add callbacks to update the LLM configuration whenever sliders change.
        - Provide immediate feedback upon updating LLM parameters.
        """
        with gr.Blocks(css=".gradio-container {font-family: 'Helvetica', sans-serif;}") as interface:
            gr.Markdown(
                """
                # Document-Aware Chatbot
                **A retrieval-augmented LLM interface.**
                
                Upload documents, ask questions, and get context-aware answers.
                """
            )

            with gr.Tabs():
                # --- Tab: Document Management ---
                with gr.Tab("Document Management"):
                    gr.Markdown("### Manage and process your documents before chatting.")
                    with gr.Row():
                        with gr.Column():
                            uploaded_files = gr.File(
                                label="Upload Documents (PDF)", file_types=[".pdf"], file_count="multiple"
                            )
                            process_button = gr.Button("Process & Store Documents")
                            process_output = gr.Textbox(label="Processing Status", interactive=False)
                        with gr.Column():
                            gr.Markdown("#### Documents in the Vector Database:")
                            docs_list = gr.Textbox(label="Stored Documents", interactive=False)
                            refresh_docs_button = gr.Button("Refresh Document List")

                    process_button.click(
                        self.chatbot_interface.process_and_store_documents,
                        inputs=[uploaded_files],
                        outputs=[process_output],
                    )

                    def list_docs():
                        docs = self.chatbot_interface.list_documents()
                        return "\n".join(docs) if docs else "No documents found."

                    refresh_docs_button.click(list_docs, inputs=[], outputs=[docs_list])

                # --- Tab: Chat Interface ---
                with gr.Tab("Chat"):
                    gr.Markdown("### Interact with the LLM using the ingested documents.")
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("#### Ask a Question")
                            query_input = gr.Textbox(
                                label="Your Question", placeholder="Ask something relevant to your documents..."
                            )
                            chat_button = gr.Button("Ask the Bot")

                            gr.Markdown("#### Bot Response")
                            response_output = gr.Textbox(label="Response", lines=8, interactive=False)

                        with gr.Column(scale=2):
                            gr.Markdown("#### Retrieved Context")
                            with gr.Accordion("Click to view retrieved context", open=False):
                                context_display = gr.Textbox(label="Context", lines=20, interactive=False)

                    chat_button.click(
                        self.chatbot_interface.chat_with_bot,
                        inputs=[query_input],
                        outputs=[context_display, response_output],
                    )

                # --- Tab: Advanced Settings ---
                with gr.Tab("Advanced Settings"):
                    gr.Markdown("### Fine-tune Model Parameters or Reset the State")

                    with gr.Group():
                        gr.Markdown("#### LLM Parameters")
                        gr.Markdown(
                            "Adjust the sliders below to change the creativity and response length of the model."
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=self.chatbot_interface.llm_config.temperature,
                            label="Response Creativity (Temperature)",
                        )
                        max_length_slider = gr.Slider(
                            minimum=50,
                            maximum=2000,
                            step=50,
                            value=self.chatbot_interface.llm_config.max_tokens,
                            label="Max Response Length (Tokens)",
                        )
                        temp_status = gr.Textbox(label="Temperature Update Status", interactive=False)
                        max_tokens_status = gr.Textbox(label="Max Tokens Update Status", interactive=False)

                        def update_temp(temp: float) -> str:
                            return self.chatbot_interface.update_temperature(temp)

                        def update_max_t(max_t: int) -> str:
                            return self.chatbot_interface.update_max_tokens(max_t)

                        temperature_slider.change(update_temp, inputs=[temperature_slider], outputs=[temp_status])
                        max_length_slider.change(update_max_t, inputs=[max_length_slider], outputs=[max_tokens_status])

                    gr.Markdown("#### Cleanup Operations:")
                    cleanup_button = gr.Button("Clear Index & Raw Folder")
                    cleanup_output = gr.Textbox(label="Cleanup Status", interactive=False)

                    def cleanup():
                        return self.chatbot_interface.clear_index_and_raw_folder()

                    cleanup_button.click(cleanup, inputs=[], outputs=[cleanup_output])

                # --- Tab: About & Instructions ---
                with gr.Tab("About"):
                    gr.Markdown(
                        """
                        ### About this App
                        This is a Retrieval-Augmented Generation (RAG) demo application.
                        
                        **How it works:**
                        1. **Upload Documents:** Add your PDF documents to build a vector database.
                        2. **Process & Store:** The application extracts embeddings and stores them.
                        3. **Ask Questions:** The LLM retrieves relevant context from the vector store and answers.
                        
                        **Use Cases:**
                        - Summarizing multiple documents.
                        - Finding specific information in a large set of PDFs.
                        
                        **Tips:**
                        - For best results, upload quality documents and ask clear questions.
                        - Adjust advanced settings for more creative or longer responses.
                        """
                    )

            self.interface = interface

    def launch(self) -> None:
        """
        Launch the Gradio interface.
        """
        self.interface.launch()
