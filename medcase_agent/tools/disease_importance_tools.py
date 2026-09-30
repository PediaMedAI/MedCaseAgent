"""Offline disease-name retrieval with citation metadata and source exclusions.

The historical function name is retained for agent compatibility. This module
does not infer importance, rarity, novelty, or any other disease classification.
"""
from __future__ import annotations

import gzip
import heapq
import json
import math
import os
import re
from collections import defaultdict
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import unquote

PUBLIC_FIELDS = ("pmcid", "pmid", "doi", "year", "diseases", "title", "journal", "pub_date", "license")
STOPWORDS = frozenset("a an and as at by for from in into of on or the to with without case report patient disease".split())


def _values(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[\n;,]+", value) if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return [str(value).strip()]


def _normalize_identifier(value, kind):
    value = unquote(str(value or "")).strip()
    if kind == "doi":
        value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", value, flags=re.I)
        return value.casefold().rstrip("/")
    if kind == "pmcid":
        match = re.search(r"(?:^|/)(?:PMC)?([0-9]+)(?:\.\d+)?(?:/|$)", value, re.I)
        return f"PMC{int(match[1])}" if match else ""
    match = re.search(r"(?:^|/)(?:PMID\s*:?\s*)?([0-9]+)(?:/|$)", value, re.I)
    return str(int(match[1])) if match else ""


def _tokens(value):
    return tuple(token for token in re.findall(r"[a-z0-9]+", str(value).casefold()) if len(token) > 1 and token not in STOPWORDS)


def _resource():
    override = os.environ.get("MEDCASE_DISEASE_INDEX")
    if override:
        path = Path(override).expanduser()
        return path / "records.jsonl.gz" if path.is_dir() else path
    return files("medcase_agent").joinpath("resources", "disease_index", "records.jsonl.gz")


def _index_key(resource):
    # Path stat invalidates the cache when an external index is replaced.
    try:
        stat = Path(str(resource)).stat()
        return str(resource), stat.st_mtime_ns, stat.st_size
    except OSError:
        return str(resource), None, None


@lru_cache(maxsize=1)
def _load_index(key):
    resource = _resource()
    records, token_sets, phrase_sets = [], [], []
    postings = defaultdict(list)
    with resource.open("rb") as binary:
        handle = gzip.GzipFile(fileobj=binary) if str(resource).endswith(".gz") else binary
        try:
            for line in handle:
                if not line.strip():
                    continue
                original = json.loads(line)
                if not isinstance(original, dict) or not isinstance(original.get("diseases"), list):
                    raise ValueError("Disease index rows must be objects with a diseases list")
                record = {field: original.get(field) for field in PUBLIC_FIELDS}
                for field in ("doi", "title", "journal", "pub_date", "license"):
                    if record[field] is not None and not isinstance(record[field], str):
                        raise ValueError("Citation metadata fields must be strings or null")
                record["pmcid"] = _normalize_identifier(original.get("pmcid", original.get("pmc_id")), "pmcid")
                record["pmid"] = _normalize_identifier(record["pmid"], "pmid") or None
                record["doi"] = _normalize_identifier(record["doi"], "doi") or None
                record["diseases"] = [name for name in original.get("diseases", []) if isinstance(name, str)]
                if not record["pmcid"]:
                    raise ValueError("Disease index contains an invalid PMCID")
                phrases = frozenset(" ".join(_tokens(name)) for name in record["diseases"])
                tokens = frozenset(token for phrase in phrases for token in phrase.split())
                position = len(records)
                for token in tokens:
                    postings[token].append(position)
                records.append(record)
                token_sets.append(tokens)
                phrase_sets.append(phrases)
        finally:
            if handle is not binary:
                handle.close()
    return records, token_sets, phrase_sets, dict(postings)


def _exclusions(case_data, exclude_pmcids, exclude_pmids, exclude_dois):
    excluded = {kind: {_normalize_identifier(item, kind) for item in _values(value)} - {""} for kind, value in (
        ("pmcid", exclude_pmcids), ("pmid", exclude_pmids), ("doi", exclude_dois)
    )}

    def collect(node):
        if not isinstance(node, dict):
            return
        for kind in excluded:
            for key in (kind, f"{kind}s", "pmc_id" if kind == "pmcid" else kind, f"exclude_{kind}s", f"related_{kind}s"):
                for value in _values(node.get(key)):
                    identifier = _normalize_identifier(value, kind)
                    if identifier:
                        excluded[kind].add(identifier)
        for key in ("source_identifiers", "related_identifiers", "metadata", "provenance", "_provenance", "related_exclusions", "exclusions", "_retrieval_exclusions"):
            child = node.get(key)
            if isinstance(child, list):
                for entry in child:
                    collect(entry)
            else:
                collect(child)

    collect(case_data)
    return excluded


def assess_disease_importance(
    diseases: Any = None,
    related_keywords: Any = None,
    case_context: str | None = None,
    top_k: int = 8,
    fetch_full_text: bool = False,
    run_llm_filter: bool = True,
    llm_model: str = "gpt-4.1",
    llm_base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    case_data: dict | None = None,
    execution_log: dict | None = None,
    exclude_pmcids=None,
    exclude_pmids=None,
    exclude_dois=None,
    **kwargs,
) -> dict:
    """Retrieve prior literature by disease names, without a classification step.

    Legacy LLM/full-text arguments are accepted but do not initiate remote calls.
    Source text, diagnoses, titles, and abstracts are never inferred from case_data.
    Supply diseases and/or related_keywords explicitly. Exclusions are applied
    before scoring and before the top-k cutoff.
    """
    queries = _values(diseases) + _values(related_keywords)
    query_phrases = {" ".join(_tokens(query)) for query in queries} - {""}
    query_tokens = {token for phrase in query_phrases for token in phrase.split()}
    if not query_tokens:
        raise ValueError("Provide diseases or related_keywords with searchable terms.")
    try:
        top_k = max(1, min(int(top_k), 20))
    except (TypeError, ValueError, OverflowError):
        top_k = 8
    excluded = _exclusions(case_data, exclude_pmcids, exclude_pmids, exclude_dois)
    try:
        records, token_sets, phrase_sets, postings = _load_index(_index_key(_resource()))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Disease index could not be loaded ({type(error).__name__}).") from error
    candidates = set()
    for token in query_tokens:
        candidates.update(postings.get(token, ()))
    ranked = []
    total = len(records)
    for position in candidates:
        record = records[position]
        if any(record.get(kind) in identifiers for kind, identifiers in excluded.items()):
            continue
        overlap = query_tokens & token_sets[position]
        exact_matches = query_phrases & phrase_sets[position]
        lexical = sum(math.log1p(total / max(1, len(postings[token]))) for token in overlap)
        score = lexical + 10.0 * len(exact_matches)
        year = record.get("year") if isinstance(record.get("year"), int) else 0
        entry = (score, year, record["pmcid"], position)
        if len(ranked) < top_k:
            heapq.heappush(ranked, entry)
        elif entry > ranked[0]:
            heapq.heapreplace(ranked, entry)
    results = []
    for score, _, _, position in sorted(ranked, reverse=True):
        record = dict(records[position])
        record.update({"paper_id": record["pmcid"], "score": round(score, 6), "matched_disease_terms": sorted(query_tokens & token_sets[position])})
        results.append(record)
    caveats = [
        "Lexical disease-name matches are candidate references, not evidence of clinical similarity or support for a claim.",
        "Verify the original publications before citing them; no rarity, novelty, or importance assessment is performed.",
        "Identifier exclusions only match identifiers present in the index; provide PMCID or PMID when DOI metadata is missing.",
    ]
    return {
        "retrieved_similar_cases": results,
        "top_relevant_papers": results,
        "citation_ready_dois": list(dict.fromkeys(record["doi"] for record in results if record["doi"])),
        "supporting_papers_missing_doi": [
            {field: record[field] for field in ("paper_id", "pmcid", "pmid", "title")}
            for record in results if not record["doi"]
        ],
        "assessment": {
            "summary": f"Retrieved {len(results)} candidate references using disease-name overlap.",
            "analysis_method": "local_lexical_retrieval",
            "caveats": caveats,
        },
        "warnings": [],
    }
