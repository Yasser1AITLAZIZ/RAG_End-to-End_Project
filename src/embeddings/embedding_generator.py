from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from embeddings.config import DEFAULT_MODEL_NAME, DEFAULT_BATCH_SIZE
from embeddings.exceptions import EmbeddingError


class EmbeddingGenerator:
    """
    Class for generating embeddings from text using Sentence Transformers.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the embedding generator with a specified model.

        Args:
            model_name (str): Name of the Sentence Transformers model to load.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingError(f"Failed to load model '{model_name}': {e}")

    def generate_embeddings(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to process.
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: Array of embeddings.
        """
        if not texts:
            raise ValueError("Input text list is empty. Provide at least one text.")

        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
