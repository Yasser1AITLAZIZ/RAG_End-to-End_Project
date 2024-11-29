import pytest
from src.vector_database.pinecone_client import PineconeClient
from src.vector_database.vector_manager import VectorManager
from src.vector_database.config import DEFAULT_INDEX_NAME, DEFAULT_DIMENSIONS


@pytest.fixture(scope="module")
def setup_vector_manager():
    PineconeClient()  # Initialize Pinecone
    return VectorManager(index_name=DEFAULT_INDEX_NAME, dimensions=DEFAULT_DIMENSIONS)


def test_upsert_and_query(setup_vector_manager):
    manager = setup_vector_manager

    # Add vectors
    vectors = [
        {"id": "vec1", "values": [0.1, 0.2, 0.3]},
        {"id": "vec2", "values": [0.4, 0.5, 0.6]},
    ]
    manager.upsert_vectors(vectors)

    # Query vectors
    query_result = manager.query_vectors([0.1, 0.2, 0.3], top_k=1)
    assert len(query_result) == 1
    assert query_result[0]["id"] == "vec1"


def test_delete_vector(setup_vector_manager):
    manager = setup_vector_manager

    # Delete vector
    manager.delete_vector("vec1")

    # Query to confirm deletion
    query_result = manager.query_vectors([0.1, 0.2, 0.3], top_k=1)
    assert len(query_result) == 0
