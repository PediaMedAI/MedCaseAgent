"""Convert local selected JATS articles into five-group clinical atoms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medcase_agent.extraction import AtomsExtractorPipeline
from medcase_agent.utils import load_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing one XML/asset subdirectory per case")
    parser.add_argument("--output", type=Path, required=True, help="Separate directory for atoms, anonymized image filenames, and provenance")
    parser.add_argument("--model", help="OpenAI-compatible model ID (default: OPENAI_MODEL)")
    parser.add_argument("--env-file", type=Path, help="Optional dotenv file; existing environment variables take precedence")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Optional number of cases to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--char-limit", type=int, default=100000)
    parser.add_argument("--include-figure-context", action="store_true", help="Allow figure captions during extraction only; captions are not copied to agent input")
    args = parser.parse_args(argv)
    try:
        load_environment(args.env_file)
        pipeline = AtomsExtractorPipeline(
            data_dir=args.input, out_dir=args.output, model_id=args.model,
            num_folders=args.limit, bs=args.workers, seed=args.seed,
            char_limit=args.char_limit,
            included_sections=["tables", "figures"] if args.include_figure_context else ["tables"],
        )
        summary = pipeline.run()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
