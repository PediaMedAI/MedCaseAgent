"""Create reference Markdown from original XML for separate human inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medcase_agent.conversion import MDConversionPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Separate reference output; do not use as the agent case directory")
    parser.add_argument("--case-ids", nargs="+", help="Optional subset of case directory names")
    parser.add_argument("--include-authors", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = MDConversionPipeline(args.input, args.output, display_authors=args.include_authors, doc_ids=args.case_ids).run()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
