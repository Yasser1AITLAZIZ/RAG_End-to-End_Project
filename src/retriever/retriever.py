from typing import List
from vector_database.vector_manager import VectorManager
from embeddings.embedding_generator import EmbeddingGenerator
from retriever.config import TOP_K_RESULTS
from retriever.exceptions import RetrieverError


class Retriever:
    """
    A class to retrieve the most relevant documents based on a query.
    """

    def __init__(
        self,
        vector_manager: VectorManager = VectorManager(),
        embedding_generator: EmbeddingGenerator = EmbeddingGenerator(),
    ):
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
            List[Dict[str, Union[str, float]]]: A list of dictionaries with keys 'id', 'score', and 'text'.
        """
        try:
            print(query)
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embeddings(texts=[query])[0]
            vec_embedding = query_embedding.astype(float).tolist()

            # Query the vector database
            results = self.vector_manager.query_vectors(query_vector=vec_embedding, top_k=top_k)

            # Include text from metadata
            return [
                {"id": match["id"], "score": float(match["score"]), "text": match["metadata"].get("text", "")}
                for match in results
            ]
        except Exception as e:
            raise RetrieverError(f"Failed to retrieve documents: {e}")
