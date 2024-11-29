from pinecone_client import PineconeClient
from vector_manager import VectorManager

if __name__ == "__main__":
    PineconeClient()

    manager = VectorManager()
    vectors = [
        {"id": "vec1", "values": [0.1, 0.2, 0.3]},
        {"id": "vec2", "values": [0.4, 0.5, 0.6]},
    ]
    manager.upsert_vectors(vectors)

    query_result = manager.query_vectors([0.2, 0.3, 0.35])
    print("Query Result:", query_result)

    # manager.delete_vector("vec1")
    # print("Vector vec1 deleted.")
