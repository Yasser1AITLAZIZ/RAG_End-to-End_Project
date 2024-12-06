from typing import List, Dict, Union
from vector_database.pinecone_client import PineconeClient
from embeddings.chunks import DocumentChunker
from embeddings.embedding_generator import EmbeddingGenerator
from vector_database.config import DEFAULT_INDEX_NAME, DEFAULT_DIMENSIONS, NAMESPACE
from vector_database.exceptions import PineconeError
import uuid


class VectorManager:
    """
    Manages vector operations in Pinecone, including creation, insertion, retrieval, and deletion.
    """

    def __init__(
        self,
        directory_documents: str,
        index_name: str = DEFAULT_INDEX_NAME,
        dimensions: int = DEFAULT_DIMENSIONS,
        namespace: str = NAMESPACE,
    ):
        """
        Initialize and configure a vector index on Pinecone.

        Args:
            directory_documents (str, optional): A directory path containing documents (if any).
            index_name (str): Name of the index to be created or used.
            dimensions (int): Dimensionality of the stored vectors.
            namespace (str): Namespace under which the vectors are organized.

        Raises:
            PineconeError: If initialization or index creation fails.
        """
        self.index_name = index_name
        self.dimensions = dimensions
        self.namespace = namespace

        try:
            # Initialize Pinecone client and create the index if not existing
            self.client = PineconeClient()
            self.client.create_index(index_name=index_name, dimensions=dimensions)
            self.index = self.client.client.Index(index_name)
        except Exception as e:
            raise PineconeError(f"Failed to initialize vector index '{index_name}': {e}")

    def upsert_vectors(self, vectors: List[Dict[str, Union[str, List[float]]]]) -> None:
        """
        Insert or update vectors in the configured Pinecone index.

        Args:
            vectors (List[Dict[str, Union[str, List[float]]]]): A list of dictionaries,
                each containing 'id' and 'values' keys, where 'values' is the vector.

        Raises:
            PineconeError: If the upsert operation fails.
        """
        try:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            print("Added vectors to the index successfully!")
        except Exception as e:
            raise PineconeError(f"Failed to upsert vectors: {e}")

    def query_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Query the index for the most similar vectors to the provided query vector.

        Args:
            query_vector (List[float]): The vector to query for similarity.
            top_k (int): Number of most similar vectors to retrieve.

        Returns:
            List[Dict[str, Union[str, float]]]: A list of matches, where each match includes
            the vector's 'id', 'score', and optionally metadata and values.

        Raises:
            PineconeError: If the query operation fails.
        """
        try:
            return self.index.query(
                vector=query_vector, namespace=self.namespace, top_k=top_k, include_metadata=True, include_values=True
            )["matches"]
        except Exception as e:
            raise PineconeError(f"Failed to query vectors: {e}")

    def fetch_vectors(self, vector_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch specified vectors from the index by their IDs.

        Args:
            vector_ids (List[str]): List of vector IDs to be retrieved.

        Returns:
            Dict[str, Dict]: A dictionary mapping vector IDs to their details.

        Raises:
            PineconeError: If the fetch operation fails.
        """
        try:
            response = self.index.fetch(ids=vector_ids, namespace=self.namespace)
            return response["vectors"]
        except Exception as e:
            raise PineconeError(f"Failed to fetch vectors: {e}")

    def delete_vector(self, vector_id: str) -> None:
        """
        Delete a vector from the index based on its ID.

        Args:
            vector_id (str): The ID of the vector to delete.

        Raises:
            PineconeError: If the deletion operation fails.
        """
        try:
            self.index.delete(ids=[vector_id], namespace=self.namespace)
        except Exception as e:
            raise PineconeError(f"Failed to delete vector '{vector_id}': {e}")

    def delete_index(self) -> None:
        """
        Delete the entire index from Pinecone.

        Raises:
            PineconeError: If the index deletion fails.
        """
        try:
            self.client.client.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted successfully!")
        except Exception as e:
            raise PineconeError(f"Failed to delete index '{self.index_name}': {e}")

    def embed_store_db(self, directory_documents: str) -> None:
        """
        Process documents by chunking them, generating embeddings using the existing EmbeddingGenerator,
        and storing them in Pinecone. This method manually handles the embeddings and metadata insertion.

        Args:
            directory_documents (str): The directory containing documents to be processed.

        Raises:
            PineconeError: If there's an issue during the embedding or indexing process.
        """
        try:
            # 1. Chunk documents
            chunker = DocumentChunker(directory=directory_documents, chunk_size=450, chunk_overlap=100)
            documents = chunker.chunk_data()  # documents should be a list of Document objects

            # 2. Extract the texts to embed
            texts = [doc.page_content for doc in documents]

            # 3. Generate embeddings for all texts
            embeddings = EmbeddingGenerator().generate_embeddings(texts=texts)

            # 4. Prepare the vectors for upsert to Pinecone
            vectors = []
            for doc, embedding in zip(documents, embeddings):
                vector_id = str(uuid.uuid4())
                # Include the document text in the metadata
                vector_entry = {
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": {**doc.metadata, "text": doc.page_content},
                }
                vectors.append(vector_entry)

            # 5. Upsert the vectors into Pinecone
            self.upsert_vectors(vectors=vectors)

        except Exception as e:
            raise PineconeError(f"Failed during embedding and storage process: {e}")
