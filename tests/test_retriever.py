import pytest
import time
from typing import List, Dict, Union
from src.vector_database.vector_manager import VectorManager
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retriever.retriever import Retriever


@pytest.fixture(scope="module")
def setup_retriever():
    """
    Fixture to set up a Retriever instance for testing.
    Creates a VectorManager, an EmbeddingGenerator, and a Retriever.
    Cleans up by deleting the index after tests.
    """
    manager = VectorManager(
        directory_documents="",  # No documents needed for this test
        index_name="pytest-retriever-test",
        dimensions=384,
        namespace="testing",
    )
    embedding_generator = EmbeddingGenerator()
    retriever = Retriever(manager, embedding_generator)
    yield retriever
    manager.delete_index()


def test_retrieve(setup_retriever):
    """
    Test the retrieve method of the Retriever class.
    Inserts vectors with metadata, queries the retriever,
    and checks the format and content of the returned results.
    """
    retriever = setup_retriever

    # Add test vectors to the database with metadata including 'text'
    vectors: List[Dict[str, Union[str, List[float], Dict]]] = [
        {
            "id": "doc1",
            "values": [0.1, 0.2, 0.3] + [0.0] * 381,
            "metadata": {"text": "This is the content of document 1."},
        },
        {
            "id": "doc2",
            "values": [0.4, 0.5, 0.6] + [0.0] * 381,
            "metadata": {"text": "This is the content of document 2."},
        },
    ]
    retriever.vector_manager.upsert_vectors(vectors)
    time.sleep(15)  # Short wait to ensure indexing is complete

    # Retrieve based on a query
    query = "Example query for document retrieval"
    results = retriever.retrieve(query)

    assert len(results) > 0, "No results retrieved."
    for result in results:
        assert "id" in result, "Result missing 'id' field."
        assert "score" in result, "Result missing 'score' field."
        assert "text" in result, "Result missing 'text' field."
        # Verify that returned text matches one of the inserted texts
        assert result["text"] in [
            "This is the content of document 1.",
            "This is the content of document 2.",
        ], "Retrieved text does not match any inserted document content."
