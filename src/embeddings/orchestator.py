from embedding_generator import EmbeddingGenerator

if __name__ == "__main__":
    texts = [
        "OptimRiskMaximizer is a new method for risk management in trading.",
        "It combines advanced mathematical optimization techniques.",
    ]
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(texts)

    print("Generated Embeddings:")
    for idx, embedding in enumerate(embeddings):
        print(f"Text {idx + 1}: {texts[idx]}")
        print(f"Embedding: {embedding[:5]}...")
