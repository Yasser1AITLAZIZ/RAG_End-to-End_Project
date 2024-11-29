from typing import Any, Dict
from .base_loader import BaseDocumentLoader
from .preprocessor import TextPreprocessor


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for plain text documents."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load and preprocess a plain text document.

        Args:
            file_path (str): The path to the text file.

        Returns:
            Dict[str, Any]: A dictionary containing the processed text and metadata.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
            processed_text = self.preprocessor.process(raw_text)
            metadata = {
                "file_path": file_path,
                "file_type": "txt",
            }
            return {"text": processed_text, "metadata": metadata}
        except Exception as e:
            raise IOError(f"Error loading text file {file_path}: {e}")
