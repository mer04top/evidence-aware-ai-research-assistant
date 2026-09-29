from app.corpus.ingest import chunk_text


def make_words(n):
    return ' '.join(f'word{i}' for i in range(n))


def test_chunk_size_is_respected():
    text = make_words(120)

    chunks = chunk_text(text, chunk_size=50, overlap=0)

    assert len(chunks[0].split()) == 50


def test_chunks_overlap_correctly():
    text = make_words(100)

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    # last 10 words of chunk 0 should reappear as the first 10 of chunk 1
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_short_text_returns_a_single_chunk():
    chunks = chunk_text('just a few words here', chunk_size=50, overlap=10)

    assert len(chunks) == 1