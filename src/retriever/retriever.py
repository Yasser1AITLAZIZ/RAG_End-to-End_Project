from typing import List
from src.vector_database.vector_manager import VectorManager
from src.embeddings.embedding_generator import EmbeddingGenerator
from retriever.config import TOP_K_RESULTS
from retriever.exceptions import RetrieverError


class Retriever:
    """
    A class to retrieve the most relevant documents based on a query.
    """

    def __init__(self, vector_manager: VectorManager, embedding_generator: EmbeddingGenerator):
        """
        Initialize the retriever.

        Args:
            vector_manager (VectorManager): The vector database manager.
            embedding_generator (EmbeddingGenerator): The embedding generator.
        """
        self.vector_manager = vector_manager
        self.embedding_generator = embedding_generator

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[dict]:
        """
        Retrieve the top K most relevant documents for a given query.

        Args:
            query (str): The query string.
            top_k (int): Number of results to retrieve.

        Returns:
            List[Dict[str, float]]: A list of document IDs and similarity scores.
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embeddings([query])[0]
            vec_embedding = query_embedding.astype(float).tolist()  # must have type list[float] tp query in Pinecone
            # Query the vector database
            results = self.vector_manager.query_vectors(query_vector=vec_embedding, top_k=top_k)

            # Format results
            return [{"id": match["id"], "score": float(match["score"])} for match in results]
        except Exception as e:
            raise RetrieverError(f"Failed to retrieve documents: {e}")
