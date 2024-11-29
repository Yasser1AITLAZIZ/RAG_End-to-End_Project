from typing import Any, Dict
from .base_loader import BaseDocumentLoader
from .preprocessor import TextPreprocessor
import docx


class WordDocumentLoader(BaseDocumentLoader):
    """Loader for Word documents."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load and preprocess a Word document.

        Args:
            file_path (str): The path to the Word file.

        Returns:
            Dict[str, Any]: A dictionary containing the processed text and metadata.
        """
        try:
            doc = docx.Document(file_path)
            raw_text = "\n".join([para.text for para in doc.paragraphs])
            processed_text = self.preprocessor.process(raw_text)
            metadata = {
                "file_path": file_path,
                "file_type": "docx",
            }
            return {"text": processed_text, "metadata": metadata}
        except Exception as e:
            raise IOError(f"Error loading Word file {file_path}: {e}")
