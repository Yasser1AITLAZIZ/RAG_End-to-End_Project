import pytest
from src.embeddings.embedding_generator import EmbeddingGenerator


def test_generate_embeddings():
    # Initialize the embedding generator
    generator = EmbeddingGenerator()

    # Test case: Valid input
    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = generator.generate_embeddings(texts)

    assert embeddings.shape[0] == len(texts), "Number of embeddings does not match number of texts."

    # Test case: Empty input
    with pytest.raises(ValueError):
        generator.generate_embeddings([])
