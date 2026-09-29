import pymupdf

def extract_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    return '\n'.join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    # word-count based - simple + good enough
    # retrieval already handles messy/missing text
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap

    return chunks


def ingest_pdf(pdf_path: str, paper_id: str) -> list[dict]:
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    return [{'id': f'{paper_id}_{i}', 'text': chunk, 'source': paper_id} for i, chunk in enumerate(chunks)]