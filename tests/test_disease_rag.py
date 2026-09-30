"""Hermetic tests: every publication below is synthetic."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from medcase_agent.tools import disease_importance_tools as rag


def record(number, disease="synthetic blue fever", year=2020, **extra):
    return {
        "pmcid": f"PMC{number}", "pmid": str(number + 100),
        "doi": f"10.9999/synthetic.{number}", "year": year,
        "diseases": [disease], "title": f"Synthetic reference {number}",
        "journal": "Synthetic Journal", "pub_date": str(year), "license": "CC0",
        **extra,
    }


@pytest.fixture
def index(tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl.gz"

    def write(rows):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        monkeypatch.setenv("MEDCASE_DISEASE_INDEX", str(path))
        rag._load_index.cache_clear()
        return path

    yield write
    rag._load_index.cache_clear()


@pytest.mark.parametrize("argument,identifier", [
    ("exclude_pmcids", "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/"),
    ("exclude_pmids", "https://pubmed.ncbi.nlm.nih.gov/101/"),
    ("exclude_dois", "https://doi.org/10.9999/SYNTHETIC.1"),
])
def test_identifiers_are_excluded_before_top_k(index, argument, identifier):
    index([record(1, year=2026), record(2, year=2020)])
    result = rag.assess_disease_importance("synthetic blue fever", top_k=1, **{argument: [identifier]})
    assert [row["pmcid"] for row in result["retrieved_similar_cases"]] == ["PMC2"]


def test_source_and_related_provenance_excluded_without_source_prose(index):
    index([record(1, year=2026), record(2, year=2025), record(3)])
    case_data = {
        "source_identifiers": {"pmcid": "PMC1"},
        "related_exclusions": {"pmids": ["102"]},
        "metadata": {"title": "SOURCE_TITLE_SENTINEL", "abstract": "SOURCE_ABSTRACT_SENTINEL"},
    }
    result = rag.assess_disease_importance("synthetic blue fever", case_data=case_data)
    assert [row["pmcid"] for row in result["retrieved_similar_cases"]] == ["PMC3"]
    assert "SENTINEL" not in json.dumps(result)
    with pytest.raises(ValueError, match="Provide diseases"):
        rag.assess_disease_importance(case_data=case_data)


def test_only_disease_names_are_searched_and_only_public_fields_returned(index):
    index([
        record(1, category="SECRET_CATEGORY", rarity_level="SECRET_RARITY", abstract="SECRET_ABSTRACT"),
        record(2, disease="unrelated diagnosis", title="synthetic blue fever"),
    ])
    result = rag.assess_disease_importance("synthetic blue fever", run_llm_filter=True, fetch_full_text=True)
    assert [row["pmcid"] for row in result["retrieved_similar_cases"]] == ["PMC1"]
    assert "SECRET" not in json.dumps(result)
    assert result["assessment"]["analysis_method"] == "local_lexical_retrieval"
    assert set(result["retrieved_similar_cases"][0]) == set(rag.PUBLIC_FIELDS) | {"paper_id", "score", "matched_disease_terms"}


def test_missing_doi_retains_traceable_pmc_and_pmid(index):
    index([record(1, doi=None)])
    result = rag.assess_disease_importance(related_keywords=["synthetic blue fever"])
    assert result["citation_ready_dois"] == []
    assert result["supporting_papers_missing_doi"][0]["pmcid"] == "PMC1"
    assert result["supporting_papers_missing_doi"][0]["pmid"] == "101"


def test_no_fabricated_results_and_bounded_top_k(index):
    index([record(1)])
    assert rag.assess_disease_importance("absent term")["retrieved_similar_cases"] == []
    assert len(rag.assess_disease_importance("synthetic blue fever", top_k="invalid")["retrieved_similar_cases"]) == 1


def test_plain_jsonl_override(tmp_path, monkeypatch):
    path = tmp_path / "custom.jsonl"
    path.write_text(json.dumps(record(1)) + "\n")
    monkeypatch.setenv("MEDCASE_DISEASE_INDEX", str(path))
    result = rag.assess_disease_importance("synthetic blue fever")
    assert result["retrieved_similar_cases"][0]["pmcid"] == "PMC1"
    rag._load_index.cache_clear()


def test_missing_index_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDCASE_DISEASE_INDEX", str(tmp_path / "missing.jsonl"))
    with pytest.raises(ValueError, match="could not be loaded"):
        rag.assess_disease_importance("synthetic blue fever")


def test_malformed_index_is_an_error(tmp_path, monkeypatch):
    path = tmp_path / "invalid.jsonl"
    path.write_text("[]\n")
    monkeypatch.setenv("MEDCASE_DISEASE_INDEX", str(path))
    with pytest.raises(ValueError, match="could not be loaded"):
        rag.assess_disease_importance("synthetic blue fever")


def test_export_allowlist_determinism_and_legacy_name_flattening(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "export_disease_index.py"
    spec = importlib.util.spec_from_file_location("disease_export", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    disease_input = tmp_path / "input.jsonl"
    disease_input.write_text(json.dumps({
        "pmc_id": 1, "year": 2020, "category": "SECRET_CATEGORY",
        "diseases": ["synthetic fever", {"Superclass": "synthetic fever", "Subtype": "synthetic blue fever", "Rare Variant": None}],
    }) + "\n")
    database = tmp_path / "metadata.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE publications (pmc_id INTEGER, pmid INTEGER, title TEXT, abstract TEXT, category TEXT, rarity_level TEXT, reasoning TEXT)")
        connection.execute("INSERT INTO publications VALUES (1, 101, 'Synthetic reference', 'SECRET_ABSTRACT', 'SECRET_CATEGORY', 'SECRET_RARITY', 'SECRET_REASONING')")
    first = module.export_index(disease_input, [database], tmp_path / "first")
    second = module.export_index(disease_input, [database], tmp_path / "second")
    assert first["resource_sha256"] == second["resource_sha256"]
    with gzip.open(tmp_path / "first" / "records.jsonl.gz", "rt") as handle:
        row = json.loads(handle.readline())
    assert set(row) == set(module.FIELDS)
    assert row["diseases"] == ["synthetic fever", "synthetic blue fever"]
    assert row["pmid"] == "101"
    assert "SECRET" not in json.dumps(row)
    assert first["record_count"] == 1


def test_export_applies_only_documented_identifier_correction(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "export_disease_index.py"
    spec = importlib.util.spec_from_file_location("disease_export_correction", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    disease_input = tmp_path / "input.jsonl"
    disease_input.write_text(json.dumps({"pmc_id": 1, "year": 2020, "diseases": ["synthetic fever"]}) + "\n")
    database = tmp_path / "metadata.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE publications (pmc_id INTEGER, pmid INTEGER)")
        connection.execute("INSERT INTO publications VALUES (1, 101)")
    corrections = tmp_path / "corrections.json"
    payload = {"corrections": [{"pmcid": "PMC1", "expected": {"pmid": "101"}, "set": {"pmid": "102"}}]}
    corrections.write_text(json.dumps(payload))
    manifest = module.export_index(disease_input, [database], tmp_path / "out", corrections)
    with gzip.open(tmp_path / "out" / "records.jsonl.gz", "rt") as handle:
        assert json.loads(handle.readline())["pmid"] == "102"
    assert manifest["corrections"] == {"file": "corrections.json", "sha256": module.sha256(corrections), "applied_count": 1}
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT pmid FROM publications").fetchone()[0] == 101
    payload["corrections"][0].update(expected={"category": None}, set={"category": "forbidden"})
    corrections.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only replace"):
        module.export_index(disease_input, [database], tmp_path / "blocked", corrections)
