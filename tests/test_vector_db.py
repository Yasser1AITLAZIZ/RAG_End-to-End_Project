import pytest
import time
from typing import Dict, List, Union
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
