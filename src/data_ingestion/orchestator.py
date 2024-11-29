from ingestion_pipeline import DataIngestionPipeline
import json
import os

if __name__ == "__main__":
    file_paths = [
        "data/raw/document1.txt",
        "data/raw/document2.pdf",
        "data/raw/document3.docx",
        "data/raw/document4.html",
    ]
    pipeline = DataIngestionPipeline()
    processed_documents = pipeline.ingest(file_paths)

    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    for idx, doc in enumerate(processed_documents):
        output_path = os.path.join(output_dir, f"document_{idx+1}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=4)
