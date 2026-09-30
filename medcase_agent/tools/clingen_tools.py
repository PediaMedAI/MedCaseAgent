import requests
import urllib.parse
from typing import Dict, Any, List

def search_clingen_by_keyword(query: str, **kwargs) -> Dict[str, Any]:
    """
    Resolves fuzzy text queries to an official ClinGen UUID via NCBI ClinVar.
    """
    print(f"\n[Tool Execution] 🔍 Resolving '{query}' via NCBI ClinVar...")

    encoded_query = urllib.parse.quote(query)
    esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&term={encoded_query}&retmode=json"

    try:
        esearch_resp = requests.get(esearch_url, timeout=10)
        esearch_resp.raise_for_status()
        esearch_data = esearch_resp.json()

        id_list = esearch_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {"status": "error", "message": f"No ClinVar variant found for query: '{query}'"}

        clinvar_id = id_list[0]
        print(f"[Tool Execution] ✓ NCBI matched query to ClinVar ID: {clinvar_id}")

        clingen_search_url = (
            "https://erepo.genome.network/evrepo/api/summary/classifications"
            f"?columns=cvId&values={clinvar_id}&matchTypes=exact"
        )

        clingen_resp = requests.get(clingen_search_url, timeout=10)
        clingen_resp.raise_for_status()
        clingen_data = clingen_resp.json()

        records = clingen_data if isinstance(clingen_data, list) else clingen_data.get("data", clingen_data.get("results", []))

        if not records or len(records) == 0:
            return {"status": "error", "message": f"Variant (ClinVar ID {clinvar_id}) exists, but lacks a ClinGen Expert Panel curation."}

        clingen_uuid = records[0].get("uuid")
        if not clingen_uuid:
            return {"status": "error", "message": "ClinGen record found, but UUID is missing."}

        return {
            "status": "success",
            "matches_found": len(records),
            "clinvar_id": clinvar_id,
            "uuid": clingen_uuid
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def extract_evidence_comments(data: Any, comments_list: List[str]):
    """Recursively searches the JSON-LD tree for expert 'comments'."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'comments' and isinstance(value, str):
                comments_list.append(value)
            else:
                extract_evidence_comments(value, comments_list)
    elif isinstance(data, list):
        for item in data:
            extract_evidence_comments(item, comments_list)

def fetch_clingen_variant_data(uuid: str, **kwargs) -> Dict[str, Any]:
    """
    Fetches the raw JSON-LD from ClinGen and parses it into a clean summary.
    """
    print(f"[Tool Execution] 📥 Fetching SEPIO data for UUID '{uuid}'...")
    api_endpoint = f"https://erepo.genome.network/evrepo/api/summary/classification/{uuid}/doc/sepio/version/1.0.0"

    try:
        response = requests.get(api_endpoint, timeout=10)
        response.raise_for_status()
        raw_json = response.json()

        data_node = raw_json.get("data", {})
        condition = data_node.get("condition", {}).get("label", "Unknown Condition")
        classification = data_node.get("statementOutcome", {}).get("label", "Unknown Classification")

        variant_node = data_node.get("variant", {}).get("relatedIdentifier", [{}])
        variant_name = variant_node[0].get("label", "Unknown Variant") if variant_node else "Unknown Variant"

        evidence_list = []
        extract_evidence_comments(data_node, evidence_list)

        return {
            "variant": variant_name,
            "condition": condition,
            "classification": classification,
            "evidence_bullets": evidence_list
        }
    except Exception as e:
        return {"error": str(e)}
