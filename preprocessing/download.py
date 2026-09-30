"""Download XML and image assets for an explicit, preselected list of PMC IDs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .xml_source import IMAGE_EXTENSIONS

BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com"


def normalize_pmcid(value: str) -> str:
    value = str(value).strip().upper()
    if not re.fullmatch(r"(?:PMC)?[1-9]\d*", value):
        raise ValueError(f"Invalid PMC identifier: {value!r}")
    return value if value.startswith("PMC") else f"PMC{value}"


def asset_url(value: str) -> str:
    """Convert official PMC S3 locations to HTTPS without changing the host."""
    parsed = urlsplit(value)
    if parsed.scheme == "s3" and parsed.netloc == "pmc-oa-opendata":
        return BASE_URL + parsed.path
    if parsed.scheme == "https" and parsed.netloc == "pmc-oa-opendata.s3.amazonaws.com":
        return value
    raise ValueError("Asset URL is outside the official PMC OA bucket")


def collect_assets(metadata: dict, *, include_pdf: bool = False) -> list[str]:
    xml = metadata.get("xml_url")
    if not isinstance(xml, str) or not xml:
        raise ValueError("PMC metadata has no XML asset")
    urls = [asset_url(xml)]
    media = metadata.get("media_urls") or []
    if not isinstance(media, list):
        raise ValueError("PMC media_urls must be a list")
    for value in media:
        if isinstance(value, str) and Path(urlsplit(value).path).suffix.lower() in IMAGE_EXTENSIONS:
            urls.append(asset_url(value))
    if include_pdf and metadata.get("pdf_url"):
        urls.append(asset_url(metadata["pdf_url"]))
    return list(dict.fromkeys(urls))


def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "MedCaseAgent-preprocessing/0.1"
    return session


def download_article(
    pmcid: str,
    output: str | Path,
    *,
    session: requests.Session | None = None,
    include_pdf: bool = False,
    overwrite: bool = False,
    version: int = 1,
) -> dict:
    """Download one selected article. A supplied session makes offline tests possible."""
    pmcid = normalize_pmcid(pmcid)
    if version < 1:
        raise ValueError("Article version must be positive")
    owns_session = session is None
    session = session or create_session()
    article_dir = Path(output) / pmcid
    article_dir.mkdir(parents=True, exist_ok=True)
    metadata_url = f"{BASE_URL}/metadata/{pmcid}.{version}.json"
    try:
        response = session.get(metadata_url, timeout=30)
        response.raise_for_status()
        metadata = response.json()
        if not isinstance(metadata, dict):
            raise ValueError("PMC metadata is not a JSON object")
        if metadata.get("is_manuscript") is True or str(metadata.get("is_manuscript", "")).lower() == "true":
            raise ValueError("PMC identifies this record as an author manuscript; select the published article version")
        urls = collect_assets(metadata, include_pdf=include_pdf)
        names: set[str] = set()
        planned: list[tuple[str, str]] = []
        for url in urls:
            name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("Invalid filename in PMC asset URL")
            if name in names:
                raise ValueError(f"Multiple PMC assets have the same filename: {name}")
            names.add(name)
            planned.append((url, name))
        # Keep source metadata in the source directory, separate from agent input.
        (article_dir / "source_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for url, name in planned:
            target = article_dir / name
            if not overwrite and target.is_file() and target.stat().st_size:
                continue
            temporary = target.with_name(target.name + ".part")
            try:
                with session.get(url, stream=True, timeout=60) as asset:
                    asset.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in asset.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                if not temporary.stat().st_size:
                    raise ValueError(f"Empty PMC asset: {name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        manifest = {"pmcid": pmcid, "version": version, "metadata_url": metadata_url, "files": [name for _, name in planned]}
        (article_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        if owns_session:
            session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmc-ids", nargs="+", default=[], help="Preselected PMC IDs, e.g. PMC12345 PMC67890")
    parser.add_argument("--pmc-id-file", type=Path, help="One preselected PMC ID per line; blank lines and # comments are ignored")
    parser.add_argument("--output", type=Path, required=True, help="Source article directory")
    parser.add_argument("--include-pdf", action="store_true", help="Also download PDFs; extraction uses the XML")
    parser.add_argument("--version", type=int, default=1, help="PMC OA article version (default: 1)")
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing files")
    args = parser.parse_args(argv)
    values = list(args.pmc_ids)
    if args.pmc_id_file:
        values.extend(line.partition("#")[0].strip() for line in args.pmc_id_file.read_text(encoding="utf-8").splitlines())
    try:
        pmcids = list(dict.fromkeys(normalize_pmcid(value) for value in values if value))
        if not pmcids:
            parser.error("Provide --pmc-ids or a nonempty --pmc-id-file")
        if args.version < 1:
            parser.error("--version must be positive")
    except ValueError as error:
        parser.error(str(error))
    failures = []
    with create_session() as session:
        for pmcid in pmcids:
            try:
                result = download_article(pmcid, args.output, session=session, include_pdf=args.include_pdf, overwrite=args.overwrite, version=args.version)
                print(f"{pmcid}: {len(result['files'])} assets ready")
            except (requests.RequestException, OSError, ValueError) as error:
                failures.append({"pmcid": pmcid, "error": str(error)})
                print(f"{pmcid}: failed: {error}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "download_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
