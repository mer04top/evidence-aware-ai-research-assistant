import pymupdf


def extract_text(pdf_path: str) -> str:
    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]

    return "\n".join(pages)

def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def ingest_pdf(pdf_path: str, paper_id: str) -> list[dict]:
    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    documents = []

    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"{paper_id}_{i}",
            "text": chunk,
            "source": paper_id
        })

    return documents
