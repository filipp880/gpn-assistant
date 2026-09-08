import pytest

from core import (
    chunk_with_overlap,
    adaptive_chunk,
    tokenize_lemmas,
    rrf_fusion,
    weighted_rrf_fusion,
    normalize_token,
    resolve_slang_terms,
    OkapiBM25,
)

DICT = {
    "ГПН": "Газпром нефть",
    "ГПНР": "Газпромнефть-Развитие",
    "НГДУ": "Нефтегазодобывающее управление",
    "КПЭ": "Ключевые показатели эффективности",
}


# --- chunk_with_overlap ---
def test_chunk_with_overlap_sizes():
    text = "а" * 1000
    chunks = chunk_with_overlap(text, chunk_size=500, overlap=150)
    assert chunks
    assert all(len(c) <= 500 for c in chunks)
    # 1000 символов при шаге 350 даёт ceil(1000/350)=3 чанка
    assert len(chunks) == 3


def test_chunk_with_overlap_empty():
    assert chunk_with_overlap("") == []
    assert chunk_with_overlap("   ") == []


# --- tokenize_lemmas ---
def test_tokenize_lemmas_basic():
    assert tokenize_lemmas("скважины")[0] == "скважина"


def test_tokenize_lemmas_regex():
    assert "нефть" in tokenize_lemmas("нефтяной; НЕФТЬ;")


# --- rrf_fusion ---
def test_rrf_fusion_common_first():
    fused = rrf_fusion([["a", "b", "c"], ["a", "c", "b"]], top_k=2)
    assert fused[0] == "a"


def test_rrf_fusion_top_k():
    fused = rrf_fusion([["a", "b", "c"]], top_k=3)
    assert len(fused) == 3


def test_rrf_fusion_empty():
    assert rrf_fusion([]) == []


# --- weighted_rrf_fusion ---
def test_weighted_rrf_equal_weights_match_rrf():
    rankings = [["a", "b", "c"], ["a", "c", "b"]]
    assert weighted_rrf_fusion(rankings, [1.0, 1.0], top_k=2) == rrf_fusion(rankings, top_k=2)


def test_weighted_rrf_high_weight_dominates():
    # у лексики (2-й список) вес 10 — её топ должен выйти первым
    fused = weighted_rrf_fusion([["a", "b", "c"], ["x", "y", "z"]], [1.0, 10.0], top_k=2)
    assert fused[0] == "x"
    assert fused[1] == "y"


def test_weighted_rrf_mismatched_weights_guard():
    fused = weighted_rrf_fusion([["a", "b"], ["b", "a"]], [7.0], top_k=2)
    assert fused[0] == "a"


# --- adaptive_chunk ---
def test_adaptive_chunk_empty():
    assert adaptive_chunk("") == []
    assert adaptive_chunk("   \n  ") == []


def test_adaptive_chunk_respects_paragraphs():
    text = "Первый абзац. Один. Два.\nВторой абзац. Три."
    chunks = adaptive_chunk(text, max_chunk=1000)
    assert chunks == ["Первый абзац. Один. Два.", "Второй абзац. Три."]


def test_adaptive_chunk_does_not_break_sentences():
    text = "Левое предложение. Правое предложение. " * 40
    chunks = adaptive_chunk(text, max_chunk=60)
    assert all(len(c) <= 60 for c in chunks)
    # при адаптивном чанкинге граница не падает на середину предложения:
    # каждый чанк — это цепочка целых предложений
    restored = " ".join(chunks).replace("  ", " ")
    assert restored.strip() == text.strip()


def test_adaptive_chunk_hard_splits_long_sentence():
    sent = "с" * 1000
    chunks = adaptive_chunk(sent, max_chunk=200)
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks) == sent


def test_adaptive_chunk_keeps_markdown_headings():
    text = "# Стратегия\n\nПервое предложение. Второе.\n\n## Инновации\n\nТретье."
    chunks = adaptive_chunk(text, max_chunk=1000)
    assert chunks[0].startswith("Стратегия\n")
    assert chunks[1].startswith("Инновации\n")
    assert "Первое предложение. Второе." in chunks[0]
    assert "Третье." in chunks[1]


def test_adaptive_chunk_keeps_plain_text_headings():
    text = ("Стратегия и цифровизация\n\nКомпания внедряет ML и RPA.\n\n"
            "Инновации\n\nВедутся НИОКР и применяются УПСВ.")
    chunks = adaptive_chunk(text, max_chunk=1000)
    assert chunks[0].startswith("Стратегия и цифровизация\n")
    assert "ML и RPA" in chunks[0]
    assert chunks[1].startswith("Инновации\n")
    assert "НИОКР и применяются УПСВ" in chunks[1]


def test_adaptive_chunk_plain_sentence_not_treated_as_heading():
    text = "Обычное предложение без точки\nСледующий абзац."
    chunks = adaptive_chunk(text, max_chunk=1000)
    # первая строка — короткая и без точки, но у неё есть сосед сверху/снизу:
    # она не отдельный абзац (нет пустой строки) -> не заголовок
    assert chunks and not chunks[0].endswith("\n")


# --- OkapiBM25 ---
def test_bm25_ranks_containing_doc_higher():
    corpus = [
        "Компания внедряет машинное обучение для цифровизации процессов.",
        "Летом на море тепло и солнечно, рыбы много.",
    ]
    bm25 = OkapiBM25(corpus)
    scores = bm25.get_scores("машинное обучение")
    assert scores[0] > scores[1]


def test_bm25_lemma_form_in_search():
    corpus = [
        "Внедряются насосные станции на месторождениях.",
        "История искусства древней Греции.",
    ]
    bm25 = OkapiBM25(corpus)
    scores = bm25.get_scores("насосной станцией на месторождении")
    assert scores[0] > scores[1]


def test_bm25_empty_corpus():
    bm25 = OkapiBM25([])
    assert bm25.get_scores("любой запрос") == []


# --- normalize_token ---
def test_normalize_token():
    assert normalize_token("ГПН-Р") == "ГПНР"
    assert normalize_token("гпн/р") == "ГПНР"


def test_normalize_token_homoglyph():
    # латинская H -> кириллическая Н
    assert normalize_token("ГПH") == "ГПН"
    # латинская A -> кириллическая А
    assert normalize_token("ГАЗПPOM") == "ГАЗПРОМ"
    # ё -> е
    assert normalize_token("СБЕРЁЖ") == "СБЕРЕЖ"


# --- resolve_slang_terms ---
def test_no_dictionary():
    q = "привет"
    assert resolve_slang_terms(q, {}) == (q, [])


def test_exact_abbrev_not_annotated():
    # Точное совпадение — не «опечатка», расшифровка не вставляется
    expanded, terms = resolve_slang_terms("ГПНР", DICT)
    assert terms == []
    assert expanded == "ГПНР"


def test_typo_abbrev_detected():
    # КПЕ -> КПЭ (одиночная замена, сходство 75% < старого порога 80)
    expanded, terms = resolve_slang_terms("КПЕ по скважинам", DICT)
    assert terms and terms[0]["canonical"] == "Ключевые показатели эффективности"
    assert "(" in expanded


def test_typo_with_separator():
    # ГПН-Р -> ГПНР
    _, terms = resolve_slang_terms("отчёт по ГПН-Р", DICT)
    assert any(t["canonical"] == "Газпромнефть-Развитие" for t in terms)


def test_lowercase_abbrev_detected():
    # "гпнр" набрано строчными — всё равно кейс-независимо распознаётся
    _, terms = resolve_slang_terms("гпнр", DICT)
    assert any(t["canonical"] == "Газпромнефть-Развитие" for t in terms)


def test_short_and_digit_tokens_skipped():
    expanded, terms = resolve_slang_terms("и 2025 3ГПН", DICT)
    # "и" (1 символ) и "2025" (цифры) игнорируются
    assert not any(t["original"] == "и" for t in terms)
    assert not any(t["original"] == "2025" for t in terms)
    # "3ГПН" -> "3ГПН" не совпадает с ключами словаря
    assert "(" not in expanded


def test_already_enriched_token_skipped():
    # Токен, уже содержащий расшифровку, повторно не обрабатывается
    expanded, _ = resolve_slang_terms("ГПНР (Газпромнефть-Развитие)", DICT)
    assert expanded.count("(") == 1


# --- обратная карта: расшифровка -> аббревиатура ---
def test_reverse_map_full_phrase():
    expanded, terms = resolve_slang_terms("Ключевые показатели эффективности компании", DICT)
    assert "КПЭ" in expanded
    assert any(t["original"] == "Ключевые показатели эффективности" and t["canonical"] == "КПЭ" for t in terms)


def test_reverse_map_inflection():
    # «показателями эффективности» (творительный/родительный падежи) -> леммы КПЭ
    expanded, _ = resolve_slang_terms("ключевыми показателями эффективности", DICT)
    assert "КПЭ" in expanded


def test_reverse_map_no_partial_phrase():
    # Без слова «ключевые» фраза не считается КПЭ — защита от ложных срабатываний
    expanded, _ = resolve_slang_terms("показатели эффективности проекта", DICT)
    assert "(" not in expanded


def test_reverse_map_abbrev_already_present():
    q = "АЗС и автозаправочная станция"
    expanded, _ = resolve_slang_terms(q, {"АЗС": "автозаправочная станция"})
    assert expanded == q


def test_empty_query():
    assert resolve_slang_terms("", DICT) == ("", [])


# --- homoglyph в распознавании ---
def test_homoglyph_latin_h_recognized():
    # «ГПН» с латинской H распознаётся как «Газпром нефть»
    expanded, terms = resolve_slang_terms("отчёт по ГПH", DICT)
    assert any(t["canonical"] == "Газпром нефть" for t in terms)
    assert "Газпром нефть" in expanded


def test_homoglyph_mixed_alphabet_gpnr():
    # «ГПНР» с наполовину латинскими буквами («ГПНР» в смешанном алфавите)
    expanded, terms = resolve_slang_terms("ГПН-P отчёт", DICT)
    assert any(t["canonical"] == "Газпромнефть-Развитие" for t in terms)
    assert "Газпромнефть-Развитие" in expanded


def test_exact_latin_abbrev_not_annotated():
    # Правильная латинская EBITDA — не опечатка, расшифровка не вставляется
    expanded, terms = resolve_slang_terms("EBITDA за 2025", DICT)
    assert terms == []
    assert "(" not in expanded


def test_ordinary_latin_word_not_false_positive():
    # Обычные латинские слова не должны случайно сопоставляться со словарём
    _, terms = resolve_slang_terms("переведи слово COVID", DICT)
    assert terms == []