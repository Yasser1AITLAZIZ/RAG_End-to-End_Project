from pinecone import Pinecone, ServerlessSpec
from vector_database.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT
from vector_database.exceptions import PineconeError


class PineconeClient:
    """
    Manages the connection to Pinecone and initializes the service.
    """

    def __init__(self):
        """
        Initialize the Pinecone client with the provided API key and environment.
        """
        try:
            self.client = Pinecone(api_key=PINECONE_API_KEY)
        except Exception as e:
            raise PineconeError(f"Failed to initialize Pinecone: {e}")

    def list_indexes(self) -> list:
        """
        List all existing indexes in Pinecone.

        Returns:
            list: A list of index names.
        """
        try:
            return self.client.list_indexes().names()
        except Exception as e:
            raise PineconeError(f"Failed to list indexes: {e}")

    def create_index(self, index_name: str, dimensions: int, metric: str = "cosine"):
        """
        Create a new index in Pinecone.

        Args:
            index_name (str): Name of the index to create.
            dimensions (int): Dimensionality of the vectors to store.
            metric (str): Metric to use for similarity search (default: cosine).
        """
        try:
            if index_name not in self.list_indexes():
                self.client.create_index(
                    name=index_name,
                    dimension=dimensions,
                    metric=metric,
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
                )
                print("Created index successfully")
        except Exception as e:
            raise PineconeError(f"Failed to create index '{index_name}': {e}")
