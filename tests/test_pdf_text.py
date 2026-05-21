from pdf_text import chunk_text


def test_chunk_text_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, words_per_chunk=100, overlap_words=10)
    assert len(chunks) > 1
    assert all(chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
