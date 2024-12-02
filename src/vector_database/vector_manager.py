from typing import List, Dict, Union
from vector_database.pinecone_client import PineconeClient
from vector_database.config import DEFAULT_INDEX_NAME, DEFAULT_DIMENSIONS, NAMESPACE
from vector_database.exceptions import PineconeError


class VectorManager:
    """
    Handles vector operations in Pinecone, including creation, insertion, retrieval, and deletion.
    """

    def __init__(
        self, index_name: str = DEFAULT_INDEX_NAME, dimensions: int = DEFAULT_DIMENSIONS, namespace: str = NAMESPACE
    ):
        """
        Initialize a vector index in Pinecone.

        Args:
            index_name (str): Name of the index to create or use.
            dimensions (int): Dimensionality of the vectors to store.
            namespace (str): Namespace for organizing vectors.
        """
        self.index_name = index_name
        self.dimensions = dimensions
        self.namespace = namespace

        try:
            # Initialize Pinecone client
            self.client = PineconeClient()
            self.client.create_index(index_name=index_name, dimensions=dimensions)
            self.index = self.client.client.Index(index_name)
        except Exception as e:
            raise PineconeError(f"Failed to initialize vector index '{index_name}': {e}")

    def upsert_vectors(self, vectors: List[Dict[str, Union[str, List[float]]]]):
        """
        Add or update vectors in the index.

        Args:
            vectors (List[Dict[str, Union[str, List[float]]]]): List of dictionaries with 'id' and 'values'.
        """
        try:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            print("Added vectors to the index successfully!")
        except Exception as e:
            raise PineconeError(f"Failed to upsert vectors: {e}")

    def query_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Query the index for the most similar vectors.

        Args:
            query_vector (List[float]): The query vector to search.
            top_k (int): Number of nearest neighbors to retrieve.

        Returns:
            List[Dict[str, Union[str, float]]]: List of matching vectors with scores.
        """
        try:
            return self.index.query(
                vector=query_vector, namespace=self.namespace, top_k=top_k, include_metadata=True, include_values=True
            )["matches"]
        except Exception as e:
            raise PineconeError(f"Failed to query vectors: {e}")

    def fetch_vectors(self, vector_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch specific vectors from the index by their IDs.

        Args:
            vector_ids (List[str]): List of vector IDs to fetch.

        Returns:
            Dict[str, Dict]: Dictionary of vectors retrieved, keyed by their IDs.

        Raises:
            PineconeError: If the fetch operation fails.
        """
        try:
            response = self.index.fetch(ids=vector_ids, namespace=self.namespace)
            return response["vectors"]
        except Exception as e:
            raise PineconeError(f"Failed to fetch vectors: {e}")

    def delete_vector(self, vector_id: str):
        """
        Delete a specific vector from the index.

        Args:
            vector_id (str): ID of the vector to delete.
        """
        try:
            self.index.delete(ids=[vector_id], namespace=self.namespace)
        except Exception as e:
            raise PineconeError(f"Failed to delete vector '{vector_id}': {e}")

    def delete_index(self):
        """
        Delete the entire index from Pinecone.

        Raises:
            PineconeError: If the delete operation fails.
        """
        try:
            self.client.client.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted successfully!")
        except Exception as e:
            raise PineconeError(f"Failed to delete index '{self.index_name}': {e}")
