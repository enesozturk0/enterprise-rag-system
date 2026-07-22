from src.ingestion.chunker import TextChunker
from src.ingestion.models import Document, DocumentMetadata


def _make_page(source: str, page: int, total_pages: int, content: str) -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(
            source=source,
            page=page,
            total_pages=total_pages,
        ),
    )


def test_chunk_respects_max_size():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    long_text = "Bu cümle oldukça uzun bir metin ve kesinlikle yirmi karakterden fazla."
    page = _make_page("test.pdf", 1, 1, long_text)

    chunks = chunker.chunk([page])

    assert all(len(c.content) <= 20 for c in chunks)


def test_chunk_indices_are_sequential_within_a_source():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    page = _make_page(
        "test.pdf",
        1,
        1,
        "Bu bir test metnidir ve birden fazla parçaya bölünecek kadar uzundur.",
    )

    chunks = chunker.chunk([page])

    indices = [c.metadata.chunk_index for c in chunks]

    # Chunk indices should start at 0 and increase without gaps.
    assert indices == list(range(len(chunks)))


def test_chunk_indices_continue_across_pages_of_the_same_source():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)

    page1 = _make_page(
        "test.pdf",
        1,
        2,
        "Birinci sayfanın metni burada uzunca yazılı.",
    )
    page2 = _make_page(
        "test.pdf",
        2,
        2,
        "İkinci sayfanın metni de burada uzunca yazılı.",
    )

    chunks = chunker.chunk([page1, page2])

    indices = [c.metadata.chunk_index for c in chunks]

    # Chunk numbering should remain continuous across page boundaries.
    assert indices == list(range(len(chunks)))


def test_chunk_inherits_metadata_from_source_page():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)

    page = _make_page(
        "report.pdf",
        3,
        10,
        "Bu metin birden fazla chunk'a bölünecek kadar uzun bir örnek.",
    )

    chunks = chunker.chunk([page])

    assert all(c.metadata.source == "report.pdf" for c in chunks)
    assert all(c.metadata.page == 3 for c in chunks)
    assert all(c.metadata.total_pages == 10 for c in chunks)


def test_chunk_index_restarts_for_each_source_document():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)

    page_a = _make_page(
        "a.pdf",
        1,
        1,
        "A dosyasının uzunca bir metni burada yazılıdır.",
    )
    page_b = _make_page(
        "b.pdf",
        1,
        1,
        "B dosyasının da uzunca bir metni burada yazılıdır.",
    )

    chunks = chunker.chunk([page_a, page_b])

    a_indices = [c.metadata.chunk_index for c in chunks if c.metadata.source == "a.pdf"]
    b_indices = [c.metadata.chunk_index for c in chunks if c.metadata.source == "b.pdf"]

    # Each source document maintains its own chunk index sequence.
    assert a_indices[0] == 0
    assert b_indices[0] == 0
