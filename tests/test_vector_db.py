import pytest
from vector_database.pinecone_client import PineconeClient
from vector_database.vector_manager import VectorManager


@pytest.fixture(scope="module")
def setup_vector_manager():
    PineconeClient()  # Initialize Pinecone
    return VectorManager(index_name="pytest", dimensions=3)


# def test_upsert_and_query(setup_vector_manager):
# manager = setup_vector_manager

# Add vectors
# vectors = [
#    {"id": "vec1", "values": [0.1, 0.2, 0.3]},
#    {"id": "vec2", "values": [0.4, 0.5, 0.6]},
# ]
# manager.upsert_vectors(vectors)
# time.sleep(1)
# Query vectors
# query_result = manager.fetch_vectors(["vec1"])

# assert len(query_result) == 1


def test_delete_vector(setup_vector_manager):
    manager = setup_vector_manager

    # Add vectors
    vectors = [
        {"id": "vec1", "values": [0.1, 0.2, 0.3]},
        {"id": "vec2", "values": [0.4, 0.5, 0.6]},
    ]
    manager.upsert_vectors(vectors)

    # Delete vector
    manager.delete_vector("vec2")

    # Query to confirm deletion
    query_result = manager.fetch_vectors(["vec2"])
    print(query_result)
    assert len(query_result) == 0

    manager.delete_index()
