"""Command-line interface for input validation, inference and local retrieval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .utils import load_environment

ATOM_FIELDS = ("history", "presentation", "diagnostics", "management", "outcome")


def validate_case(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Case JSON does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Case JSON must contain an object.")
    fields = (*ATOM_FIELDS, "diagnosis")
    if not any(data.get(field) for field in ATOM_FIELDS):
        raise ValueError("Provide at least one nonempty clinical atom group.")
    for field in fields:
        value = data.get(field, [])
        if not isinstance(value, (str, list)):
            raise ValueError(f"{field} must be a string or a list of strings.")
        if isinstance(value, list) and not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field} must contain strings only.")
    if "metadata" in data and not isinstance(data["metadata"], dict):
        raise ValueError("metadata must be an object when supplied.")
    provenance = path.parent / "provenance.json"
    if provenance.exists() and not isinstance(json.loads(provenance.read_text()), dict):
        raise ValueError("provenance.json must contain an object.")
    return {"case": str(path), "atom_groups": [key for key in fields if data.get(key)],
            "provenance": provenance.is_file(), "status": "valid"}


def read_exclusions(path):
    if path is None:
        return {"pmcids": [], "pmids": [], "dois": []}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {"pmcids", "pmids", "dois"}:
        raise ValueError("Exclusions must be an object with pmcids, pmids and/or dois arrays.")
    for field in ("pmcids", "pmids", "dois"):
        data.setdefault(field, [])
        if not isinstance(data[field], list) or not all(isinstance(item, str) for item in data[field]):
            raise ValueError(f"Exclusions field {field} must be a list of strings.")
    return data


def parser():
    p = argparse.ArgumentParser(description="MedCaseAgent preprocessing and report-writing inference.")
    p.add_argument("--env", type=Path, help="Dotenv file (default: .env in the current directory).")
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate clinical atoms without an API call.")
    validate.add_argument("case", type=Path)
    run = sub.add_parser("generate", help="Run Planner → Writer → Editor and final audit.")
    run.add_argument("case", type=Path, help="An atoms JSON file; optional images live in its sibling imgs/.")
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--model", help="Override OPENAI_MODEL.")
    run.add_argument("--disease-index", type=Path)
    run.add_argument("--exclude-ids", type=Path, help="JSON containing related pmcids, pmids and/or dois to exclude.")
    run.add_argument("--validate-only", action="store_true", help="Check the input and exclusions without an API call.")
    rag = sub.add_parser("rag", help="Search the bundled Disease-Index RAG corpus locally.")
    rag.add_argument("diseases", nargs="+")
    rag.add_argument("--index", type=Path)
    rag.add_argument("--exclude-ids", type=Path)
    rag.add_argument("--top-k", type=int, default=5)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        load_environment(args.env)
        if args.command == "validate":
            print(json.dumps(validate_case(args.case), indent=2))
            return 0
        if args.command == "rag":
            from .tools.disease_importance_tools import assess_disease_importance

            if args.index:
                os.environ["MEDCASE_DISEASE_INDEX"] = str(args.index.resolve())
            exclusions = read_exclusions(args.exclude_ids)
            result = assess_disease_importance(
                diseases=args.diseases, top_k=args.top_k,
                exclude_pmcids=exclusions["pmcids"],
                exclude_pmids=exclusions["pmids"],
                exclude_dois=exclusions["dois"],
            )
            if result.get("error"):
                raise RuntimeError(result["error"])
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        checked = validate_case(args.case)
        exclusions = read_exclusions(args.exclude_ids)
        if args.disease_index:
            if not args.disease_index.is_file():
                raise ValueError(f"Disease index does not exist: {args.disease_index}")
            os.environ["MEDCASE_DISEASE_INDEX"] = str(args.disease_index.resolve())
        if args.validate_only:
            print(json.dumps(checked, indent=2))
            return 0
        model = args.model or os.getenv("OPENAI_MODEL", "")
        if not model or model.startswith("your-"):
            raise ValueError("Set OPENAI_MODEL or pass --model with your provider's model identifier.")
        from .generation import GenerationPipeline

        pipeline = GenerationPipeline(
            working_dir=str(args.output.resolve()), model_id=model, mode="multi",
            tools_config={"exclusions": exclusions},
        )
        result = pipeline.process_case(str(args.case.resolve()))
        if result.get("status") != "success":
            raise RuntimeError(result.get("error_message") or "Report generation failed.")
        print(json.dumps({key: result.get(key) for key in ("status", "output_file", "log_path")}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        p.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
