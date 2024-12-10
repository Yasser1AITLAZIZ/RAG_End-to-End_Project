import signal
import sys
from ui.chatbot import ChatbotInterface
from ui.frontend import Frontend

# Initialize the chatbot interface
chatbot_interface = ChatbotInterface()


def cleanup_on_exit(signal_received, frame):
    """
    Clean up resources when the application is terminated.
    """
    print("Shutting down the application... Cleaning vector database and raw data folder.")
    chatbot_interface.clear_index_and_raw_folder()
    sys.exit(0)


# Register the signal handlers
signal.signal(signal.SIGINT, cleanup_on_exit)
signal.signal(signal.SIGTERM, cleanup_on_exit)

# Build and launch the interface
frontend = Frontend(chatbot_interface)
frontend.build_interface()

if __name__ == "__main__":
    frontend.launch(share=True, server_port=7861)
