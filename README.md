# MedCaseAgent

[Project page & walkthrough](https://huggingface.co/spaces/PediaMedAI/MedCaseAgent)
· MedCase-150K
· [MedCase-Bench](https://huggingface.co/datasets/PediaMedAI/MedCase-Bench)

Code for **MedCaseAgent: From Multimodal Clinical Evidence to Rare Disease Case
Reports**. Given clinical atoms and images, the **Planner** organizes evidence
and figures, the **Writer** drafts the report, and the **Editor** checks clinical
fidelity, citations and structure. Final audit and citation-repair passes reuse
the retrieved evidence. Tools cover PubMed literature, medical images, ClinGen
and a local Disease-Index RAG corpus.

This release contains preprocessing, report-generation inference and the disease
retrieval index.

## Setup

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `.env` for your OpenAI-compatible provider:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your-vision-language-model
```

Use a model that supports image input and native function calling. Set the base
URL and model name to those of your provider. Shell environment variables take
precedence over `.env`; credentials are never bundled. Optional NCBI and image
tool settings are described in [.env.example](.env.example).

Composite-image inspection uses the configured vision model. For specialized
MedGemma inspection, configure `MEDCASE_MEDGEMMA_BASE_URL` and
`MEDCASE_MEDGEMMA_MODEL`, or install `pip install -e '.[local-vision]'` and set
`MEDCASE_MEDGEMMA_MODEL` to an accessible model ID/local directory.

## Run

An input JSON contains `history`, `presentation`, `diagnostics`, `management`
and `outcome`, each a list of clinical facts. Place images in an `imgs/` folder
beside the JSON, without source captions. A deliberately incomplete synthetic
input is provided for checking the installation:

```bash
# Offline validation; no API calls.
medcase-agent validate examples/synthetic_case/case_atoms.json

# Run on your own clinical atoms and images; makes model and retrieval calls.
medcase-agent generate work/cases/PMC123456/PMC123456_atoms.json --output runs
```

Each run writes a Markdown report, stage outputs, tool traces and images under
`runs/<case_id>/`. The same commands work as `python -m medcase_agent ...`.
Use `--env path/to/.env` before the command to load a different API configuration.

## Preprocessing

Start with a list of **preselected PMC Open Access articles** or local JATS XML.
Download the XML and images, extract patient-specific clinical atoms, and keep
the source article and reference report separate from the agent input:

```text
Selected PMC IDs → JATS XML + images → clinical atoms + uncaptioned images
                                    ↘ provenance and optional reference report
```

See [Preprocessing](docs/preprocessing.md) for commands, directory layouts,
extraction fields and validation. The article-selection and disease-classification
stages are outside this release; these commands do not reconstruct the full
paper dataset or its benchmark split by themselves.

## Disease-Index RAG

The disease index is bundled with the package and can be queried without an API:

```bash
medcase-agent rag "Erdheim-Chester disease" --top-k 5
```

The index contains disease names and public article identifiers/metadata, with
classification labels removed. See [Disease index](docs/disease_index.md) for
the exact record count, schema and source information. Set
`MEDCASE_DISEASE_INDEX` or use `--disease-index` on `generate` for another index.

For source-derived inputs, preprocessing writes source identifiers to
`provenance.json`; generation uses them to exclude the source from retrieval.
Supply related articles and evaluation exclusions explicitly:

```bash
medcase-agent generate work/cases/PMC123456/PMC123456_atoms.json \
  --exclude-ids examples/exclusions.json --output runs
```

Fill the JSON arrays with the actual PMCID/PMID/DOI exclusions. The example is
empty; no benchmark exclusion list is implied. Prefer PMCID and PMID when an
index record has no DOI.

## Tests and license

```bash
pip install -e '.[dev]'
pytest
```

The tests use synthetic inputs and mocked APIs. Code and original index
annotations follow the manuscript's [CC BY-NC 4.0](LICENSE) license; source
publications retain their recorded licenses. Generated reports require expert
review before use.
