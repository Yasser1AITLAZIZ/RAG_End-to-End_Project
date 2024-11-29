from typing import Any, Dict
from .base_loader import BaseDocumentLoader
from .preprocessor import TextPreprocessor
from bs4 import BeautifulSoup


class HTMLDocumentLoader(BaseDocumentLoader):
    """Loader for HTML documents."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load and preprocess an HTML document.

        Args:
            file_path (str): The path to the HTML file.

        Returns:
            Dict[str, Any]: A dictionary containing the processed text and metadata.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")
            raw_text = soup.get_text()
            processed_text = self.preprocessor.process(raw_text)
            metadata = {
                "file_path": file_path,
                "file_type": "html",
                "title": soup.title.string if soup.title else "",
            }
            return {"text": processed_text, "metadata": metadata}
        except Exception as e:
            raise IOError(f"Error loading HTML file {file_path}: {e}")
