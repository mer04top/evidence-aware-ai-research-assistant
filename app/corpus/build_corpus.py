import json
import os

from app.corpus.ingest import ingest_pdf

PAPERS_DIR = 'data/papers'
OUTPUT_PATH = 'data/corpus.json'


def build():
    documents = []

    for filename in sorted(os.listdir(PAPERS_DIR)):
        if not filename.endswith('.pdf'):
            continue

        paper_id = filename.replace('.pdf', '')
        documents.extend(ingest_pdf(os.path.join(PAPERS_DIR, filename), paper_id))
        print(f'{filename}: {len(documents)} chunks so far')

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(documents, f, indent=2)

    print(f'wrote {len(documents)} total chunks to {OUTPUT_PATH}')


if __name__ == '__main__':
    build()