"""Тесты хранилища корпоративного словаря: JSON и CSV (загрузка/сохранение)."""

import csv
import os

import pytest

import dictionary_store
from dictionary_store import load_dictionary, save_dictionary


SAMPLE = {
    "ГПН": "Газпром нефть",
    "ГПНР": "Газпромнефть-Развитие",
    "АЗС, дочка": "Автозаправочная станция, дочернее общество",
}

CSV_TEXT = (
    "term,definition\n"
    "ГПН,Газпром нефть\n"
    '"АЗС, дочка","Автозаправочная станция, дочернее общество"\n'
)


@pytest.fixture()
def tmp_json(tmp_path):
    return str(tmp_path / "dict.json")


@pytest.fixture()
def tmp_csv(tmp_path):
    return str(tmp_path / "dict.csv")


def test_is_csv():
    assert dictionary_store.is_csv("a.CSV")
    assert dictionary_store.is_csv("a.csv")
    assert not dictionary_store.is_csv("a.json")
    assert not dictionary_store.is_csv("")


def test_save_load_json_roundtrip(tmp_json):
    save_dictionary(SAMPLE, tmp_json)
    assert load_dictionary(tmp_json) == SAMPLE


def test_save_load_csv_roundtrip(tmp_csv):
    save_dictionary(SAMPLE, tmp_csv)
    assert load_dictionary(tmp_csv) == SAMPLE
    with open(tmp_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["term", "definition"]
    assert ["АЗС, дочка", "Автозаправочная станция, дочернее общество"] in rows


def test_load_csv_existing_file(tmp_path):
    path = tmp_path / "dict.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    assert load_dictionary(str(path)) == {
        "ГПН": "Газпром нефть",
        "АЗС, дочка": "Автозаправочная станция, дочернее общество",
    }


def test_load_missing_file_returns_empty():
    assert load_dictionary("no_such_file.json") == {}
    assert load_dictionary("no_such_file.csv") == {}


def test_load_csv_ignores_corrupt_rows(tmp_path):
    path = tmp_path / "dict.csv"
    path.write_text(
        "term,definition\n"
        "ГПН,Газпром нефть\n"
        "only-term\n"
        "\n"
        ",пустая_аббревиатура\n"
        "  ,  \n",
        encoding="utf-8",
    )
    assert load_dictionary(str(path)) == {"ГПН": "Газпром нефть"}


def test_load_broken_csv_returns_empty(tmp_path):
    path = tmp_path / "dict.csv"
    path.write_text("term,definition\n" + '"unclosed,row\n', encoding="utf-8")
    result = load_dictionary(str(path))
    assert result in ({}, {"term": "unclosed"})
    assert isinstance(result, dict)


def test_load_broken_json_returns_empty():
    path = os.path.join(os.path.dirname(__file__), "broken.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    try:
        assert load_dictionary(path) == {}
    finally:
        os.remove(path)


def test_agent_loads_csv_dictionary(monkeypatch, tmp_path):
    import agent

    csv_path = tmp_path / "agent_dict.csv"
    save_dictionary({"QWE": "Расшифровка QWE"}, str(csv_path))

    monkeypatch.setattr(agent, "DICTIONARY_FILE", str(csv_path))

    inst = agent.GpnAgent()
    assert inst.dictionary == {"QWE": "Расшифровка QWE"}


def test_agent_save_dictionary_writes_csv(monkeypatch, tmp_path):
    import agent

    csv_path = tmp_path / "agent_dict.csv"
    monkeypatch.setattr(agent, "DICTIONARY_FILE", str(csv_path))

    agent.save_dictionary({"TRM": "Значение термина"})
    assert load_dictionary(str(csv_path)) == {"TRM": "Значение термина"}