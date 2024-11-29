from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load and preprocess the document.

        Args:
            file_path (str): The path to the document file.

        Returns:
            Dict[str, Any]: A dictionary containing the processed text and metadata.
        """
        pass
