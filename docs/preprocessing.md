# Preprocessing

Prepare **preselected PMC Open Access articles** as clinical atoms and images
for MedCaseAgent. Article selection, disease classification and training are
outside this release. The pipeline follows the paper's input construction:
keep patient-specific observations and remove background literature,
reference-driven claims and final diagnostic exposition.

```text
Selected PMC IDs → source XML + figures → five clinical atom groups + images
                                      ↘ provenance / optional reference report
```

## 1. Install and configure

From the repository root:

```bash
pip install -e .
cp .env.example .env
```

Set these values in `.env` for atom extraction:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_MODEL=your-model-id
```

Only extraction calls a model. Downloading uses public PMC endpoints; reference
conversion is local. Shell variables take precedence over `.env`. Use
`--env-file path/to/config.env` or `--model MODEL` on the extraction command for
explicit overrides. The API compatibility options in `.env.example` also apply
here. These commands need no local training stack or model checkpoint.

For text extraction with a local Qwen3.5-9B server, install a compatible vLLM
build using the [official recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html),
then start the server:

```bash
vllm serve Qwen/Qwen3.5-9B --served-model-name Qwen3.5-9B \
  --host 127.0.0.1 --port 8000 --max-model-len 32768 \
  --reasoning-parser qwen3 --language-model-only
```

Use `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`,
`OPENAI_API_KEY=EMPTY`, and `OPENAI_MODEL=Qwen3.5-9B` in `.env`.
The reasoning parser separates internal reasoning from the final JSON response.
For report generation with images and tools, use the vision and function-calling
configuration described in the main README.

## 2. Obtain source XML and images

Create `selected_pmcids.txt` with one actual PMC ID per line; blank lines and
`#` comments are accepted. Then run:

```bash
python -m preprocessing.download \
  --pmc-id-file selected_pmcids.txt \
  --output work/source
```

Alternatively, pass IDs with `--pmc-ids`. The downloader obtains `xml_url` and
`media_urls` from official PMC OA S3 metadata. `--include-pdf` also downloads the
PDF, but extraction uses XML. See the [official PMC AWS documentation](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/)
for the service and version layout.

`--version` defaults to **1**, an explicit version rather than “latest”. Version
1 is not available for every article; use `--version 2` where appropriate and
separate commands/output roots for different versions. Explicit author-manuscript
records are rejected. Use your reviewed article list to determine suitability
and retain the returned source metadata and article-license information.

Existing nonempty assets are reused; `--overwrite` downloads them again.
Interrupted `.part` files are removed. `download_failures.json` lists failures,
and the command exits nonzero when any selected article fails. It does not
substitute different articles.

You can skip downloading and arrange existing files in this layout:

```text
work/source/
  case-a/
    article.xml
    figure1.jpg
    figure2.tif
  case-b/
    article.nxml
    imgs/
      figure.png
```

Each immediate child directory is one case. XML/NXML files must be directly
inside it; images may also be in `imgs/`, `images/` or `media/`. Keep one intended
source article per case. When several XML files are present, the reader tries
them in filename order until it finds a usable JATS article. Abstract-only
records are rejected. PDF/OCR, DICOM and video input are not handled here.

## 3. Extract clinical atoms

```bash
python -m preprocessing.extract \
  --input work/source \
  --output work/cases \
  --workers 1
```

| Option | Meaning |
| --- | --- |
| `--model MODEL` | Override `OPENAI_MODEL`. |
| `--env-file FILE` | Read an explicit dotenv file. |
| `--workers N` | Concurrent case requests; default 1. |
| `--limit N` | Sample up to N cases. |
| `--seed N` | Seed for sampling and image shuffling; default 42. |
| `--char-limit N` | Maximum clinical-text characters per request; default 100000. |
| `--include-figure-context` | Allow captions during extraction only; default off. |

The XML reader keeps clinical body paragraphs and tables. It removes front
matter, bibliography and inline reference/image links, and skips sections
identified as background, introduction, discussion, conclusion or final
diagnostic exposition. The [extraction prompt](../medcase_agent/extraction.py)
then compresses the retained text into fragmented patient-specific facts.

The output has five arrays of strings, for example:

```json
{
  "history": ["No prior respiratory illness recorded"],
  "presentation": ["Cough for two days"],
  "diagnostics": ["Temperature 37.4 C", "Small radiographic opacity"],
  "management": ["Supportive treatment"],
  "outcome": ["Symptoms resolved at seven-day follow-up"]
}
```

This example is synthetic. Empty groups are allowed; an entirely empty result
is rejected. Extra metadata fields and the legacy sixth `diagnosis` field are
not emitted. The validator checks types and removes exact duplicate facts; it
does not establish clinical correctness.

Section filtering can omit patient facts located only in an abstract or
Discussion, and unfamiliar headings can leave explanatory prose in the input.
Review atoms against the source, including chronology, negation, units and
unsupported diagnostic claims. The cleaned public prompt is not a claim to
reproduce every historical dataset record or the paper's human review process.

## 4. Output and source separation

```text
work/cases/
  case-a/
    case-a_atoms.json
    imgs/
      image_001.png
      image_002.png
    provenance.json
    atoms_execution_log.json
```

Images are shuffled with a fixed seed, assigned neutral filenames and converted
to RGB PNG with EXIF orientation applied. Only the first frame of multiframe
images is retained. Source captions and explicit atom-to-image links are not
supplied to the agent. These transformations do not remove text already visible
in image pixels.

`provenance.json` records source identifiers, the XML hash, model/prompt version,
input truncation, image mappings and conventional manuscript headings in source
order. It is an audit sidecar: the agent reads its source identifiers for
retrieval exclusion and its conventional headings for structure, while original
filenames, source prose and image mappings stay out of model prompts.

Source XML, downloaded metadata and reference reports remain in separate
directories. The log records status, timing and token usage without raw prompts
or responses. Failed cases produce a nonzero batch exit. Reruns call the model
again; old successful atoms may remain after a failed rerun, so check the latest
status log or use a fresh output directory.

After reviewing a case, run the agent on its atoms JSON:

```bash
medcase-agent generate work/cases/case-a/case-a_atoms.json --output runs
```

For evaluation, also provide related-publication and holdout identifiers using
`--exclude-ids`. See [Disease-Index RAG](disease_index.md) for the exclusion
schema. This release does not include a preset benchmark split.

## 5. Optional reference Markdown

```bash
python -m preprocessing.reference \
  --input work/source \
  --output work/references
```

This local command retains source title, abstract, body, captions and references
for inspection. It writes `work/references/<case>/<case>_gt.md` and linked images
under `reference_images/`. Options include `--include-authors` and
`--case-ids case-a case-b`. It handles simple JATS tables and figures; publisher
layout and PDF export are not reproduced. Missing images become comments.
Keep this reference output separate from the clinical input sent to the agent.

## 6. Offline example and tests

The supplied XML is invented and has placeholder identifiers. Inspect reference
conversion without credentials or network access:

```bash
mkdir -p work/synthetic_source/synthetic-case
cp examples/synthetic_article.xml work/synthetic_source/synthetic-case/article.xml
python -m preprocessing.reference \
  --input work/synthetic_source --output work/synthetic_references

pip install -e '.[dev]'
pytest -q tests/test_preprocessing.py
```

The XML-only example has no accompanying image, so a missing-image comment is
expected. Tests create synthetic images and mock model/HTTP calls to check
filtering, namespace handling, atom schema, API options, PNG conversion,
provenance separation, download resumption and interrupted-file cleanup.
