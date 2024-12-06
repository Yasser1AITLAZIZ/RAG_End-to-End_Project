import pytest
import time
import numpy as np
from typing import Dict, List, Union
from langchain.schema import Document
from vector_database.vector_manager import VectorManager
from vector_database.exceptions import PineconeError


class TestVectorManager:
    """
    Unit tests for the VectorManager class.
    """

    @pytest.fixture(scope="class")
    def vector_manager(self) -> VectorManager:
        """
        Fixture to initialize and clean up the VectorManager instance for testing.

        Yields:
            VectorManager: An instance of VectorManager for use in tests.
        """
        # Initialize the VectorManager with a test index
        manager = VectorManager(index_name="pytest-index", dimensions=3, namespace="testing")
        yield manager
        # Cleanup: delete the test index after all tests in this class have run
        manager.delete_index()

    def test_delete_vector(self, vector_manager: VectorManager) -> None:
        """
        Test the delete_vector method.

        Args:
            vector_manager (VectorManager): The VectorManager instance provided by the fixture.
        """
        # Upsert vectors into the index
        vectors: List[Dict[str, Union[str, List[float]]]] = [
            {"id": "vec1", "values": [0.1, 0.2, 0.3]},
            {"id": "vec2", "values": [0.4, 0.5, 0.6]},
        ]
        vector_manager.upsert_vectors(vectors)

        # Delete 'vec2' from the index
        vector_manager.delete_vector("vec2")
        time.sleep(1)  # Wait for deletion to propagate

        # Attempt to fetch the deleted vector
        fetched_vector: Dict[str, Dict] = vector_manager.fetch_vectors(["vec2"])

        # Assertions to verify that 'vec2' has been deleted
        assert len(fetched_vector) == 0, "'vec2' should have been deleted"

    def test_delete_index(self) -> None:
        """
        Test the delete_index method and ensure it properly deletes the index.
        """
        # Initialize a new VectorManager instance for deleting the index
        manager = VectorManager(index_name="pytest-index-to-delete", dimensions=3, namespace="test-deleting")
        manager.delete_index()

        # Attempt to fetch a vector from the deleted index, expecting an exception
        with pytest.raises(PineconeError):
            manager.fetch_vectors(["vec1"])

    def test_embed_store_db(self, vector_manager: VectorManager, mocker) -> None:
        """
        Test the embed_store_db method by mocking the DocumentChunker and EmbeddingGenerator.
        Ensures documents are embedded and stored without error.
        """
        # Mock the DocumentChunker to return a known chunked document
        mock_doc = Document(page_content="Test content", metadata={"source": "test_file.pdf"})
        mocker.patch("embeddings.chunks.DocumentChunker.chunk_data", return_value=[mock_doc])

        # Mock the EmbeddingGenerator to return a known embedding
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mocker.patch(
            "embeddings.embedding_generator.EmbeddingGenerator.generate_embeddings", return_value=mock_embedding
        )

        # Call embed_store_db
        vector_manager.embed_store_db(directory_documents="fake_directory")

        # Wait a bit for upsert to complete
        time.sleep(15)

        # Query the index with a vector close to the mocked embedding
        # We expect to find our chunked document
        query_vector = [0.1, 0.2, 0.3]
        results = vector_manager.query_vectors(query_vector=query_vector, top_k=1)

        # Check that we got at least one match and it has the metadata we expect
        assert len(results) > 0, "No vectors returned from query, expected at least one."
        metadata = results[0].get("metadata", {})
        assert metadata.get("source") == "test_file.pdf", "Metadata from the stored chunk does not match expected."
