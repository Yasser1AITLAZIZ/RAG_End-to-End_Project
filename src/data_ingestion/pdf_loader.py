from typing import Any, Dict
from .base_loader import BaseDocumentLoader
from .preprocessor import TextPreprocessor
import PyPDF2


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF documents."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load and preprocess a PDF document.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            Dict[str, Any]: A dictionary containing the processed text and metadata.
        """
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                raw_text = ""
                for page in reader.pages:
                    raw_text += page.extract_text()
            processed_text = self.preprocessor.process(raw_text)
            metadata = {
                "file_path": file_path,
                "file_type": "pdf",
                "num_pages": len(reader.pages),
            }
            return {"text": processed_text, "metadata": metadata}
        except Exception as e:
            raise IOError(f"Error loading PDF file {file_path}: {e}")
