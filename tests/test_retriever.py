import pytest
from src.vector_database.vector_manager import VectorManager
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retriever.retriever import Retriever
import time


@pytest.fixture(scope="module")
def setup_retriever():
    vector_manager = VectorManager(index_name="pytest-retriever-test", dimensions=384, namespace="testing")
    embedding_generator = EmbeddingGenerator()
    return Retriever(vector_manager, embedding_generator)


def test_retrieve(setup_retriever):
    retriever = setup_retriever

    # Add test vectors to the database
    vectors = [
        {"id": "doc1", "values": [0.1, 0.2, 0.3] + [0.0] * 381},  # Padding to 384 dimensions
        {"id": "doc2", "values": [0.4, 0.5, 0.6] + [0.0] * 381},
    ]
    retriever.vector_manager.upsert_vectors(vectors)
    time.sleep(20)
    # Retrieve based on a query
    query = "Example query for document retrieval"
    results = retriever.retrieve(query)

    assert len(results) > 0, "No results retrieved."
    assert all("id" in result and "score" in result for result in results), "Invalid result format."

    retriever.vector_manager.delete_index()
