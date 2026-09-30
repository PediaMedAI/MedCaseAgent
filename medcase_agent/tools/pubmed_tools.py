import os
import requests
import re
from typing import Union, List
import xml.etree.ElementTree as ET
from .disease_importance_tools import _normalize_identifier

def _extract_element_text(element) -> str:
    """Safely flattens XML text, including nested tags."""
    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part and part.strip()).strip()

def fetch_pubmed_details(
    pmids: Union[str, int, List[Union[str, int]]],
    timeout: int = 10,
    **kwargs
) -> List[dict]:
    """
    Fetches structured PubMed metadata for one or more PMIDs.

    Returned keys:
    - pmid
    - pmcid
    - doi
    - title
    - journal
    - year
    - abstract
    """
    if isinstance(pmids, (str, int)):
        pmids = [pmids]

    clean_pmids = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]
    if not clean_pmids:
        return []

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key = os.environ.get("NCBI_API_KEY")
    fetch_url = f"{base_url}/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(clean_pmids),
        "retmode": "xml"
    }

    if api_key:
        fetch_params["api_key"] = api_key

    response = requests.get(fetch_url, params=fetch_params, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    records_by_pmid = {}

    for article in root.findall(".//PubmedArticle"):
        pmid = _extract_element_text(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue

        title = _extract_element_text(article.find(".//ArticleTitle"))
        journal = _extract_element_text(article.find(".//Journal/Title"))

        pub_year = _extract_element_text(article.find(".//PubDate/Year"))
        if not pub_year:
            pub_year = _extract_element_text(article.find(".//PubDate/MedlineDate"))
            year_match = re.search(r"\b(19|20)\d{2}\b", pub_year)
            pub_year = year_match.group(0) if year_match else ""

        abstract_parts = []
        for abstract_text in article.findall(".//AbstractText"):
            label = abstract_text.attrib.get("Label")
            text = _extract_element_text(abstract_text)
            if not text:
                continue
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts).strip()

        doi = ""
        pmcid = ""
        for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            id_type = article_id.get("IdType")
            value = _extract_element_text(article_id)
            if id_type == "doi" and value:
                doi = value
            elif id_type == "pmc" and value:
                pmcid = value if value.startswith("PMC") else f"PMC{value}"

        records_by_pmid[pmid] = {
            "pmid": pmid,
            "pmcid": pmcid or None,
            "doi": doi or None,
            "title": title or None,
            "journal": journal or None,
            "year": int(pub_year) if str(pub_year).isdigit() else None,
            "abstract": abstract or None,
        }

    return [records_by_pmid[pmid] for pmid in clean_pmids if pmid in records_by_pmid]

def _normalized_id(value, kind):
    return _normalize_identifier(value, kind)


def _excluded(record, exclusions):
    return any(
        _normalized_id(record.get(kind), kind) in ({
            _normalized_id(value, kind) for value in (exclusions.get(f"exclude_{kind}s") or [])
        } - {""})
        for kind in ("pmcid", "pmid", "doi") if record.get(kind)
    )


def fetch_ama_citations(dois: Union[str, List[str]], **kwargs) -> str:
    """Resolve DOI references while excluding source and related publications."""
    if isinstance(dois, str):
        dois = [dois]
    headers = {"Accept": "text/x-bibliography; style=american-medical-association"}
    citations = []
    for doi in dois:
        clean_doi = _normalized_id(doi, "doi")
        if _excluded({"doi": clean_doi}, kwargs):
            continue
        if not re.fullmatch(r"10\.\d{4,9}/\S+", clean_doi):
            continue
        try:
            response = requests.get(f"https://doi.org/{clean_doi}", headers=headers, timeout=20)
            response.raise_for_status()
            citation = re.sub(r"^\[?\d+\]?\.?\s*", "", response.text.strip())
            citations.append(f"{len(citations) + 1}. {citation}")
        except requests.RequestException:
            # Errors are not reference entries and cannot become verified citations.
            continue
    return "\n\n".join(citations)


def search_pubmed(query: str, max_results: int = 5, execution_log: dict = None, **kwargs) -> str:
    """Search PubMed, removing source/related identifiers before returning prose."""
    limit = max(1, min(int(max_results), 100))
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": limit + 30, "sort": "date"}
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=20
        )
        response.raise_for_status()
        pmids = response.json().get("esearchresult", {}).get("idlist", [])
        pmids = [pmid for pmid in pmids if not _excluded({"pmid": pmid}, kwargs)]
        records = fetch_pubmed_details(pmids, timeout=20) if pmids else []
        records = [record for record in records if not _excluded(record, kwargs)][:limit]
        return "\n---\n".join(
            f"Title: {record['title']}\nJournal: {record['journal']} ({record['year']})\n"
            f"PMID: {record['pmid']} | PMCID: {record['pmcid']} | DOI: {record['doi']}\n"
            f"Abstract: {record['abstract'] or 'No abstract available.'}\n"
            for record in records
        ) or "No eligible PubMed results found."
    except (requests.RequestException, ValueError, ET.ParseError) as exc:
        return f"PubMed search failed ({type(exc).__name__})."
