from vector_manager import VectorManager

if __name__ == "__main__":
    manager = VectorManager()
    manager.embed_store_db(directory_documents="src/data/raw/")
