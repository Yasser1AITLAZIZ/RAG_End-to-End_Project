from typing import List, Dict, Any
import os
from data_ingestion.text_loader import TextDocumentLoader
from data_ingestion.pdf_loader import PDFDocumentLoader
from data_ingestion.word_loader import WordDocumentLoader
from data_ingestion.html_loader import HTMLDocumentLoader


class DataIngestionPipeline:
    """Pipeline to ingest and preprocess documents."""

    def __init__(self):
        self.loaders = {
            ".txt": TextDocumentLoader(),
            ".pdf": PDFDocumentLoader(),
            ".docx": WordDocumentLoader(),
            ".html": HTMLDocumentLoader(),
            ".htm": HTMLDocumentLoader(),
        }

    def ingest(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents.

        Args:
            file_paths (List[str]): A list of file paths to ingest.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with processed text and metadata.
        """
        documents = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            loader = self.loaders.get(ext)
            if loader:
                try:
                    document = loader.load(file_path)
                    documents.append(document)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                print(f"Unsupported file type: {file_path}")
        return documents
