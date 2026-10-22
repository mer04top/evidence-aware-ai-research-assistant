import json
import os

from app.corpus.ingest import ingest_pdf

PAPERS_DIR = "data/papers"
OUTPUT_PATH = "data/corpus.json"

def build():
    documents = []

    if not os.path.isdir(PAPERS_DIR):
        raise FileNotFoundError(f"Paper directory not found: {PAPERS_DIR}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    for filename in sorted(os.listdir(PAPERS_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PAPERS_DIR, filename)
        paper_id = os.path.splitext(filename)[0]

        paper_documents = ingest_pdf(pdf_path, paper_id)
        documents.extend(paper_documents)

        print(
            f"{filename}: {len(paper_documents)} chunks "
            f"({len(documents)} total)"
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(documents)} total chunks to {OUTPUT_PATH}")

if __name__ == "__main__":
    build()
