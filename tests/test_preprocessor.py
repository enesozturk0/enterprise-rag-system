from src.ingestion.preprocessor import TextPreprocessor


def test_join_hyphenated_words():
    p = TextPreprocessor()
    assert p.clean("kitap oku-\nmayı seviyorum") == "kitap okumayı seviyorum"


def test_normalize_whitespace_collapses_spaces_and_blank_lines():
    p = TextPreprocessor()
    result = p.clean("Merhaba    dünya\n\n\n\nnasılsın")
    assert result == "Merhaba dünya\n\nnasılsın"


def test_turkish_characters_survive_control_char_stripping():
    p = TextPreprocessor()
    text_with_control_char = "Türkçe ğüşıöç metin\x00\x1f"
    result = p.clean(text_with_control_char)
    assert "ğüşıöç" in result
    assert "\x00" not in result and "\x1f" not in result


def test_known_headers_are_removed():
    p = TextPreprocessor(known_headers_footers={"GİZLİDİR"})
    result = p.clean("GİZLİDİR\nGerçek içerik burada")
    assert "GİZLİDİR" not in result
    assert "Gerçek içerik burada" in result


def test_detect_repeated_lines_finds_header_across_pages():
    pages = [
        "GİZLİDİR\nSayfa 1 içeriği burada",
        "GİZLİDİR\nSayfa 2 içeriği burada",
        "GİZLİDİR\nSayfa 3 içeriği burada",
    ]
    detected = TextPreprocessor.detect_repeated_lines(pages, scan_lines=1, min_repeat_ratio=0.8)
    assert "GİZLİDİR" in detected


def test_detect_repeated_lines_ignores_unique_body_text():
    pages = ["Farklı içerik A", "Farklı içerik B", "Farklı içerik C"]
    detected = TextPreprocessor.detect_repeated_lines(pages, scan_lines=1, min_repeat_ratio=0.8)
    assert detected == set()


def test_single_page_never_flags_headers():
    detected = TextPreprocessor.detect_repeated_lines(["tek sayfa metni"])
    assert detected == set()