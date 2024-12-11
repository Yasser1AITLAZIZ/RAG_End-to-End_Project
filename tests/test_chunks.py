from embeddings.chunks import DocumentChunker


def test_chunk_data():
    """Test the chunk_data method of the DocumentChunker class."""
    chunker = DocumentChunker(directory="src/data/raw/document2.pdf", chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_data()

    # Test that the returned chunks are a list
    assert isinstance(chunks, list)

    # Test that each chunk is a string
    for chunk in chunks:
        assert isinstance(chunk, str)

    # Test that the length of each chunk is within the expected range
    for chunk in chunks:
        assert 40 <= len(chunk) <= 50  # Because of overlap, chunks might slightly vary
