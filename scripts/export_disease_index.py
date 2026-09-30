#!/usr/bin/env python3
"""Export disease names and an explicit allowlist of public citation metadata."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

FIELDS = ("pmcid", "pmid", "doi", "year", "diseases", "title", "journal", "pub_date", "license")
METADATA_FIELDS = ("pmid", "doi", "title", "journal", "pub_date", "license")
CORRECTION_FIELDS = frozenset({"pmid", "doi"})
EMPTY_NAMES = {"", "none", "null", "n/a", "na", "unknown", "not applicable"}


def disease_names(value):
    """Flatten legacy disease-name containers without exporting their labels."""
    names = []
    for item in value if isinstance(value, list) else []:
        candidates = [item] if isinstance(item, str) else (
            [item.get(key) for key in ("Superclass", "Subtype", "Rare Variant")]
            if isinstance(item, dict) else []
        )
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            name = " ".join(candidate.split())
            if name.casefold() not in EMPTY_NAMES and name not in names:
                names.append(name)
    return names


def canonical_pmcid(value):
    text = str(value).strip().upper().removeprefix("PMC")
    if not text.isdigit() or int(text) < 1:
        raise ValueError("Input contains an invalid PMCID")
    return f"PMC{int(text)}"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_index(diseases_path, metadata_paths, output_dir, corrections_path=None):
    diseases_path, output_dir = Path(diseases_path), Path(output_dir)
    records = {}
    input_rows = 0
    with diseases_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            input_rows += 1
            pmcid = canonical_pmcid(item.get("pmcid", item.get("pmc_id")))
            if pmcid in records:
                raise ValueError("Duplicate PMCID in disease input; resolve it explicitly")
            year = item.get("year")
            if not isinstance(year, int) or isinstance(year, bool):
                raise ValueError("Disease input years must be integers")
            records[pmcid] = {
                "pmcid": pmcid, "pmid": None, "doi": None, "year": year,
                "diseases": disease_names(item.get("diseases")),
                "title": None, "journal": None, "pub_date": None, "license": None,
            }
    sources = [{"file": diseases_path.name, "sha256": sha256(diseases_path), "role": "disease_names"}]
    for metadata_path in map(Path, metadata_paths):
        # immutable avoids sidecar writes; use a frozen, quiescent source snapshot.
        uri = metadata_path.resolve().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            columns = {r[1] for r in connection.execute("PRAGMA table_info(publications)")}
            id_field = "pmcid" if "pmcid" in columns else "pmc_id"
            if id_field not in columns:
                raise ValueError("Metadata must contain a publications table with PMCID")
            safe_columns = [field for field in METADATA_FIELDS if field in columns]
            sql = "SELECT " + ",".join([id_field] + safe_columns) + " FROM publications"
            for row in connection.execute(sql):
                try:
                    pmcid = canonical_pmcid(row[0])
                except ValueError:
                    continue
                record = records.get(pmcid)
                if record is None:
                    continue
                for field, value in zip(safe_columns, row[1:]):
                    if value is not None and str(value).strip() and not record[field]:
                        record[field] = str(value).strip()
        sources.append({"file": metadata_path.name, "sha256": sha256(metadata_path), "role": "citation_metadata"})

    correction_summary = None
    if corrections_path is not None:
        corrections_path = Path(corrections_path)
        payload = json.loads(corrections_path.read_text(encoding="utf-8"))
        applied_count = 0
        for correction in payload["corrections"]:
            pmcid = canonical_pmcid(correction["pmcid"])
            if pmcid not in records:
                raise ValueError("Correction PMCID is absent from the disease index")
            expected, replacement = correction["expected"], correction["set"]
            if not replacement or not set(replacement) <= CORRECTION_FIELDS or set(expected) != set(replacement):
                raise ValueError("Corrections may only replace explicitly matched PMID/DOI fields")
            record = records[pmcid]
            for field, value in replacement.items():
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Corrected identifiers must be nonempty strings")
                if record[field] != expected[field]:
                    raise ValueError("Correction precondition does not match the source metadata")
                record[field] = value.strip()
            applied_count += 1
        correction_summary = {
            "file": corrections_path.name, "sha256": sha256(corrections_path),
            "applied_count": applied_count,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "records.jsonl.gz"
    fd, temporary = tempfile.mkstemp(prefix=".records-", dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as binary, gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as compressed:
            for pmcid in sorted(records, key=lambda value: int(value[3:])):
                compressed.write((json.dumps(records[pmcid], ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    years = Counter(str(record["year"]) for record in records.values())
    manifest = {
        "schema_version": "1.0", "resource": target.name,
        "record_count": len(records), "input_row_count": input_rows,
        "fields": list(FIELDS), "sources": sources,
        "corrections": correction_summary,
        "resource_sha256": sha256(target), "resource_bytes": target.stat().st_size,
        "year_counts": dict(sorted(years.items())),
        "disease_name_count": sum(len(record["diseases"]) for record in records.values()),
        "unique_disease_name_count": len({name for record in records.values() for name in record["diseases"]}),
        "records_without_disease_names": sum(not record["diseases"] for record in records.values()),
        "metadata_coverage": {field: sum(bool(record[field]) for record in records.values()) for field in METADATA_FIELDS},
        "benchmark_split": None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diseases", required=True, type=Path)
    parser.add_argument("--metadata-db", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--corrections", type=Path, help="Optional documented PMID/DOI corrections with exact expected values")
    args = parser.parse_args()
    manifest = export_index(args.diseases, args.metadata_db, args.output_dir, args.corrections)
    print(json.dumps({key: manifest[key] for key in ("record_count", "disease_name_count", "resource_bytes", "resource_sha256")}))


if __name__ == "__main__":
    main()
