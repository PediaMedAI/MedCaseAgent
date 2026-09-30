# Disease reference index

The release includes the complete available disease-retrieval snapshot: **152,295
unique PMC article records**, dated 2000–2026. It is a literature lookup index,
not the full MedCase150K dataset, a release of patient records, or a benchmark
split. No train/dev/test membership or preset benchmark exclusions are included.

The compressed resource is packaged at
`medcase_agent/resources/disease_index/records.jsonl.gz` (11,204,800 bytes).
`manifest.json` beside it records the exact field allowlist, row counts, year
counts, metadata coverage, source hashes, and resource SHA-256:

```text
51f7a6f3a594778303dee58f750e95be6bf69bf88c5c6c3315e377d8a677240b
```

## Contents and provenance

Each UTF-8 JSONL row contains only these fields:

| Field | Meaning and coverage |
| --- | --- |
| `pmcid` | Canonical PMC identifier; all 152,295 rows, unique |
| `pmid` | PubMed identifier as a string; all rows |
| `doi` | DOI when available; **null in every bundled row** |
| `year` | Integer year from the disease extraction snapshot |
| `diseases` | A list of extracted disease-name strings |
| `title` | Public article title; all rows |
| `journal` | Public journal metadata; all rows |
| `pub_date` | Original publication-date metadata string; all rows |
| `license` | Source article license recorded in the metadata snapshot; all rows |

There are **442,024 disease-name entries** and **136,934 distinct strings** after
within-record deduplication and whitespace normalization. One row has no usable
disease name after empty placeholders are removed; it remains in the exported
snapshot for completeness and is not searchable by disease terms.

Disease names come from the archived `extracted_diseases.jsonl` in the earlier
`MedCaseAgent/resources/prior_case_index` bundle. That file and the separately
retained `re/dbs/extracted_diseases.jsonl` copy are byte-identical, with SHA-256
`5ac3dfc88f85a41013da9f74ab2de51503bf85044027502774e49993ef257d34`.
The obsolete original runtime location was absent when this release was made.
The complete available snapshot was used, without a sample-size cutoff.

All records join by PMCID to the historical `llm_filtered.db` citation metadata
snapshot. The exporter reads only the allowed bibliographic columns; it does
not copy classification fields from that database. The metadata source has no
DOI column, so no DOI is invented or fetched over the network. The much larger
`source_metadata.db` and annual title/abstract databases are not needed at
runtime or included in this release.

One documented bibliographic correction is applied after the metadata join:
`PMC11139362` has PMID `38828255`, replacing the legacy value `38828236`.
The corrected identifier is recorded in the MedCase-150K 2024 publication
catalog and its accompanying correction note. The package's `corrections.json`
contains this single correction and its expected original value; `manifest.json`
records that file's SHA-256 and an applied count of one. No source database is
edited. The correction mechanism permits only explicit PMID/DOI replacements,
and rejects an unexpected original value or any other field.

Some legacy disease lists stored names inside labeled objects. Export flattens
only their disease-name values and removes the labels, empty placeholders, and
duplicate names. The released data contains no disease hierarchy, category,
rarity/importance labels, reasoning, classifier responses, classification
algorithm, abstracts, full text, images, local file paths, or private cases.
The `license` value describes the article metadata snapshot; it does not replace
the source publication's terms.

## Retrieval

The legacy `assess_disease_importance` name remains callable for compatibility.
Its implementation now performs **offline lexical retrieval over disease names**
and returns candidate references. It does not classify a disease or call an LLM.
Legacy model/full-text options are accepted but have no effect. Ranking combines
inverse-frequency-weighted disease-token overlap with an exact disease-phrase
bonus, using year and PMCID for deterministic tie-breaking. Titles and abstracts
are not searched. A lexical match is not evidence of clinical similarity,
importance, rarity, novelty, or support for a manuscript claim.

```python
from medcase_agent.tools.disease_importance_tools import assess_disease_importance

result = assess_disease_importance(
    diseases=["pulmonary embolism"],
    related_keywords=["postoperative"],
    top_k=5,
    exclude_pmcids=["PMC123456"],
    exclude_pmids=["12345678"],
    exclude_dois=["10.1234/example"],
)
```

The response preserves `retrieved_similar_cases`, `top_relevant_papers`,
`citation_ready_dois`, `supporting_papers_missing_doi`, and an `assessment`
containing a retrieval summary and caveats. It contains no classification
levels. The bundled data yields an empty `citation_ready_dois`; PMCID and PMID
still identify every reference. Missing/corrupt index files and an empty query
raise `ValueError`; a valid query with no matching records returns empty lists.

The first query loads the gzip resource and builds an in-memory token index;
later queries reuse it. Installation does not download databases or models.
To use an independent index with the same schema, set:

```bash
export MEDCASE_DISEASE_INDEX=/path/to/records.jsonl.gz
# Uncompressed JSONL and a directory containing records.jsonl.gz also work.
```

Without this environment variable, the Python package resource is used, so
retrieval also works from an installed wheel and outside the source checkout.

## Source and related-publication exclusions

Pass all known identifiers for the current source and related publications.
PMCID, PMID, and DOI inputs accept their common URL forms; DOI comparisons are
case-insensitive. Matching records are removed **before scoring and before the
top-k cutoff**, so excluded references do not displace eligible results.
`case_data` can additionally carry identifier-only provenance and related
exclusions. Its source title, abstract, diagnoses, and prose are not used to
infer a query and are not echoed in the response. The generation pipeline
injects its source and supplied related-publication exclusions into tool calls.

Exclusion is based on identifiers actually present in the index. In particular,
**DOI-only exclusion cannot identify a bundled row whose DOI is null**; always
provide its PMCID or PMID as well. Different article identifiers representing
related work must each be supplied. The library cannot infer an entire benchmark
holdout or a publication family from one source identifier. For evaluation,
prepare the relevant source/related/holdout identifier list explicitly; this
snapshot does not claim to be prefiltered for any benchmark.

## Re-export and offline checks

The export script takes a frozen JSONL disease snapshot and one or more frozen
SQLite `publications` metadata snapshots. It opens SQLite read-only without
writing sidecars, selects only allowed columns, writes deterministic gzip, and
records source checksums. Earlier metadata inputs take priority; later inputs
fill missing fields. No original database is changed.

```bash
python scripts/export_disease_index.py \
  --diseases /path/to/extracted_diseases.jsonl \
  --metadata-db /path/to/llm_filtered.db \
  --corrections medcase_agent/resources/disease_index/corrections.json \
  --output-dir medcase_agent/resources/disease_index

python -m pytest -q tests/test_disease_rag.py
```

The tests use only synthetic records and a temporary SQLite database. They cover
all three exclusion identifiers before the top-k cutoff, source-prose isolation,
the export allowlist, legacy name flattening, deterministic gzip, missing DOI,
custom indexes, missing-index errors, and the documented correction mechanism.
They do not contact external services.
