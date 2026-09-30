"""Offline checks for source/agent separation and reproducible preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import requests
from PIL import Image

from medcase_agent.conversion import article_to_markdown
from medcase_agent.extraction import ATOM_GROUPS, AtomsExtractorPipeline, normalize_atoms
from preprocessing.download import BASE_URL, collect_assets, download_article, normalize_pmcid
from preprocessing.xml_source import clinical_text, manuscript_sections, parse_article, source_identifiers

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_article.xml"
ATOMS = {
    "history": ["No prior respiratory illness recorded"],
    "presentation": ["Cough for two days"],
    "diagnostics": ["Temperature 37.4 C", "Small radiographic opacity"],
    "management": ["Supportive treatment"],
    "outcome": ["Symptoms resolved at seven-day follow-up"],
}


@pytest.fixture
def local_source(tmp_path):
    source = tmp_path / "source" / "synthetic-case"
    source.mkdir(parents=True)
    (source / "article.xml").write_bytes(FIXTURE.read_bytes())
    Image.new("L", (4, 4), color=64).save(source / "synthetic-image.tif")
    return source


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


def test_clinical_prompt_excludes_source_exposition_and_captions():
    article = parse_article(FIXTURE)
    text, headings = clinical_text(article, ["tables", "authors", "year", "citations"])
    assert "cough lasting two days" in text
    assert "Temperature | 37.4 C" in text
    assert headings == ["Case presentation"]
    for forbidden in ["ABSTRACT_ONLY_TEXT", "BACKGROUND_ONLY_TEXT", "DISCUSSION_ONLY_TEXT", "DIAGNOSIS_EXPOSITION_ONLY_TEXT", "REFERENCE_ONLY_TEXT", "CAPTION_ONLY_TEXT", "Figure 1", "10.0000/synthetic-only"]:
        assert forbidden not in text
    assert source_identifiers(article) == {"pmcid": "PMC999999999", "pmid": "999999998", "doi": "10.0000/synthetic-only"}
    assert manuscript_sections(article) == ["Abstract", "Background", "Case Presentation", "Discussion", "References"]


def test_namespaced_jats_is_supported(tmp_path):
    path = tmp_path / "namespaced.xml"
    path.write_text(FIXTURE.read_text().replace('<article xmlns:xlink=', '<article xmlns="urn:jats" xmlns:xlink='), encoding="utf-8")
    text, _ = clinical_text(parse_article(path))
    assert "cough lasting two days" in text


@pytest.mark.parametrize("heading", [
    "Case 1", "Case Report", "1.1. Case Summary", "Patient information",
    "Clinical findings", "Diagnostic assessment", "Therapeutic intervention",
    "Follow-up and outcomes",
])
def test_clinical_subsection_nested_in_introduction_is_recovered(heading):
    article = ET.fromstring(f"""<article>
      <front><article-meta><title-group><article-title>TITLE_ONLY_TEXT</article-title></title-group>
        <abstract><p>ABSTRACT_ONLY_TEXT</p></abstract></article-meta></front>
      <body><sec sec-type="intro"><title>1. Introduction</title>
        <p>BACKGROUND_ONLY_TEXT</p>
        <sec><title>Objectives</title><p>OBJECTIVES_ONLY_TEXT</p></sec>
        <sec><title>{heading}</title><p>The patient had a two-day cough.</p>
          <sec><title>Final diagnosis</title><p>DIAGNOSIS_EXPOSITION_ONLY_TEXT</p></sec>
        </sec>
      </sec><sec><title>Discussion</title><p>DISCUSSION_ONLY_TEXT</p></sec></body>
      <back><ref-list><ref><p>REFERENCE_ONLY_TEXT</p></ref></ref-list></back>
    </article>""")
    text, headings = clinical_text(article, included_sections=["tables"])
    assert text == "The patient had a two-day cough."
    assert headings == [heading]


@pytest.mark.parametrize("attribute", ["", ' sec-type="cases"'])
def test_case_introduction_is_a_clinical_heading(attribute):
    article = ET.fromstring(f"""<article><body>
      <sec{attribute}><title>2. Case introduction</title><p>Observed clinical fact.</p></sec>
      <sec sec-type="cases"><title>Case report discussion</title><p>DISCUSSION_ONLY_TEXT</p></sec>
    </body></article>""")
    assert clinical_text(article)[0] == "Observed clinical fact."


def test_empty_case_recovers_clinical_paragraphs_misnested_under_ethics():
    article = ET.fromstring("""<article><body><sec sec-type="cases">
      <title>Case presentation</title><sec><title>Ethics statement</title>
        <p>The institutional review board approved this study.</p>
        <p>Informed consent for publication was obtained.</p>
        <p>The patient developed a cough after two days.</p>
        <p>Supportive treatment was given, and symptoms resolved.</p>
        <p>The authors declare no funding or competing interests.</p>
        <sec><title>Discussion</title><p>DISCUSSION_ONLY_TEXT</p></sec>
        <sec><title>Final diagnosis</title><p>DIAGNOSIS_EXPOSITION_ONLY_TEXT</p></sec>
        <fig><caption><p>CAPTION_ONLY_TEXT</p></caption></fig>
        <ref-list><ref><p>REFERENCE_ONLY_TEXT</p></ref></ref-list>
      </sec></sec></body></article>""")
    text, headings = clinical_text(article)
    assert text == "The patient developed a cough after two days.\n\nSupportive treatment was given, and symptoms resolved."
    assert headings == ["Case presentation"]


def test_ethics_recovery_is_disabled_when_standard_clinical_text_exists():
    article = ET.fromstring("""<article><body><sec sec-type="cases">
      <title>Case presentation</title><p>Observed clinical fact.</p>
      <sec><title>Ethics statement</title><p>ETHICS_SECTION_ONLY_TEXT</p></sec>
    </sec></body></article>""")
    assert clinical_text(article)[0] == "Observed clinical fact."


@pytest.mark.parametrize("heading", ["Dear Editor", "Letter"])
def test_intro_attribute_does_not_remove_letter_case_body(heading):
    article = ET.fromstring(f"""<article><body><sec sec-type="intro">
      <title>{heading}</title><p>The patient reported a two-day cough.</p>
    </sec></body></article>""")
    assert clinical_text(article)[0] == "The patient reported a two-day cough."


def test_only_tables_referenced_by_retained_clinical_content_are_loaded():
    article = ET.fromstring("""<article>
      <front><article-meta><abstract><p><xref ref-type="table" rid="abstract-table">Table A</xref></p></abstract></article-meta></front>
      <body>
        <sec><title>Introduction</title><p><xref ref-type="table" rid="intro-table">Table B</xref></p></sec>
        <sec><title>Case report</title>
          <p>Individual measurements are in <xref ref-type="table" rid="case-table">Table C</xref>.</p>
          <p>Repeated citation <xref ref-type="table" rid="case-table">Table C</xref>.</p>
          <p>A figure follows.<fig><caption><p>FIGURE_ONLY_TEXT <xref ref-type="table" rid="figure-table">Table D</xref></p></caption></fig></p>
        </sec>
        <sec><title>Discussion</title><p><xref ref-type="table" rid="discussion-table">Table E</xref></p></sec>
      </body>
      <floats-group>
        <table-wrap id="case-table"><caption><p>Observed measurements</p></caption><table>
          <tr><th>Characteristic</th><th>Case 1</th><th>Case 2</th></tr>
          <tr><td>Age (years)</td><td>40</td><td>50</td></tr>
        </table></table-wrap>
        <table-wrap id="abstract-table"><caption><p>ABSTRACT_TABLE_ONLY_TEXT</p></caption></table-wrap>
        <table-wrap id="intro-table"><caption><p>INTRO_TABLE_ONLY_TEXT</p></caption></table-wrap>
        <table-wrap id="figure-table"><caption><p>FIGURE_TABLE_ONLY_TEXT</p></caption></table-wrap>
        <table-wrap id="discussion-table"><caption><p>DISCUSSION_TABLE_ONLY_TEXT</p></caption></table-wrap>
        <table-wrap id="uncited-table"><caption><p>UNCITED_TABLE_ONLY_TEXT</p></caption></table-wrap>
      </floats-group>
    </article>""")
    text, _ = clinical_text(article, ["tables"])
    assert text.count("Clinical table:") == 1
    assert "Characteristic | Case 1 | Case 2\nAge (years) | 40 | 50" in text
    assert "ONLY_TEXT" not in text
    without_tables, _ = clinical_text(article, [])
    assert "Clinical table:" not in without_tables
    assert "Observed measurements" not in without_tables


def test_nested_case_table_reference_is_recovered_without_intro_table():
    article = ET.fromstring("""<article><body><sec><title>Introduction</title>
      <p><xref ref-type="table" rid="background">Background table</xref></p>
      <sec><title>Patient information</title><p><xref ref-type="table" rid="observations">Patient table</xref></p></sec>
      </sec></body><floats-group>
        <table-wrap id="background"><caption><p>BACKGROUND_ONLY_TEXT</p></caption></table-wrap>
        <table-wrap id="observations"><table><tr><th>Case 1</th></tr><tr><td>Observed clinical fact.</td></tr></table></table-wrap>
      </floats-group></article>""")
    text, _ = clinical_text(article)
    assert text == "Clinical table:\nCase 1\nObserved clinical fact."


def test_embedded_figures_do_not_bypass_paragraph_filtering():
    article = ET.fromstring("""<article><body><sec><title>Case report</title>
      <p>Observed clinical fact.<fig><label>Figure 7</label><caption><p>CAPTION_ONLY_TEXT</p></caption></fig> Subsequent clinical fact.</p>
    </sec></body></article>""")
    assert clinical_text(article, ["tables"])[0] == "Observed clinical fact. Subsequent clinical fact."
    with_figures, _ = clinical_text(article, ["figures"])
    assert with_figures.count("CAPTION_ONLY_TEXT") == 1
    assert "Image finding context: CAPTION_ONLY_TEXT" in with_figures
    assert "Figure 7" not in with_figures


def test_embedded_table_is_rendered_once_with_cell_boundaries():
    article = ET.fromstring("""<article><body><sec><title>Case report</title>
      <p>Observed clinical fact.<table-wrap><caption><p>Measurement</p></caption><table>
        <tr><th>Case 1</th><th>Case 2</th></tr><tr><td>10</td><td>20</td></tr>
      </table></table-wrap> Subsequent clinical fact.</p>
    </sec></body></article>""")
    text, _ = clinical_text(article, ["tables"])
    assert text == "Observed clinical fact. Subsequent clinical fact.\n\nClinical table:\nMeasurement\nCase 1 | Case 2\n10 | 20"
    assert clinical_text(article, [])[0] == "Observed clinical fact. Subsequent clinical fact."


def test_administrative_recovery_does_not_flatten_embedded_figures():
    article = ET.fromstring("""<article><body><sec sec-type="cases"><title>Case report</title>
      <sec><title>Ethics statement</title><p>Informed consent was obtained.</p>
        <p>Observed clinical fact.<fig><caption><p>CAPTION_ONLY_TEXT</p></caption></fig> Subsequent clinical fact.</p>
      </sec></sec></body></article>""")
    assert clinical_text(article, ["tables"])[0] == "Observed clinical fact. Subsequent clinical fact."


@pytest.mark.parametrize("body", [
    '<sec><title>Introduction</title><p>BACKGROUND_ONLY_TEXT</p><sec><title>Objectives</title><p>OBJECTIVES_ONLY_TEXT</p></sec></sec>',
    '<sec><title>Discussion</title><sec><title>Case report</title><p>DISCUSSION_ONLY_TEXT</p></sec></sec>',
    '<sec sec-type="cases"><title>Case presentation</title><sec><title>Ethics statement</title><p>Informed consent was obtained.</p></sec></sec>',
    '<sec><title>Ethics statement</title><p>UNSCOPED_ETHICS_ONLY_TEXT</p></sec>',
    '<sec sec-type="cases"><title>Case presentation</title><sec><title>Ethics and Discussion</title><p>DISCUSSION_ONLY_TEXT</p></sec></sec>',
])
def test_structural_recovery_does_not_open_nonclinical_sections(body):
    article = ET.fromstring(f"<article><body>{body}</body></article>")
    with pytest.raises(ValueError, match="No clinical body paragraphs"):
        clinical_text(article)


def test_extraction_outputs_only_atoms_and_unlabeled_pngs(local_source, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_SEND_TEMPERATURE", "false")
    monkeypatch.setenv("OPENAI_MAX_TOKENS_PARAM", "max_completion_tokens")
    payload = dict(ATOMS, metadata={"article_title": "must not reach atoms"}, diagnosis=["legacy conclusion"])
    client = FakeClient(json.dumps(payload))
    output = tmp_path / "prepared"
    pipeline = AtomsExtractorPipeline(local_source.parent, output, model_id="mock-model", client=client)
    assert pipeline.run() == {"total": 1, "success": 1, "failed": 0}
    case = output / local_source.name
    assert json.loads((case / "synthetic-case_atoms.json").read_text()) == ATOMS
    provenance = json.loads((case / "provenance.json").read_text())
    assert provenance["source_identifiers"]["doi"] == "10.0000/synthetic-only"
    assert provenance["image_mapping"] == {"imgs/image_001.png": "synthetic-image.tif"}
    assert provenance["input_truncated"] is False
    with Image.open(case / "imgs" / "image_001.png") as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
    assert not list(case.glob("*.xml"))
    assert not list(case.glob("*_gt.md"))
    assert "temperature" not in client.calls[0]
    assert "max_completion_tokens" in client.calls[0]
    assert "ABSTRACT_ONLY_TEXT" not in client.calls[0]["messages"][0]["content"]
    log = json.loads((case / "atoms_execution_log.json").read_text())
    assert "raw_prompt" not in log and "raw_response" not in log


def test_extraction_rejects_invalid_model_json_without_partial_atoms(local_source, tmp_path):
    pipeline = AtomsExtractorPipeline(local_source.parent, tmp_path / "bad", model_id="mock-model", client=FakeClient('{"history": "not an array"}'))
    assert pipeline.run()["failed"] == 1
    assert not list((tmp_path / "bad").rglob("*_atoms.json"))
    log = json.loads((tmp_path / "bad" / "synthetic-case" / "atoms_execution_log.json").read_text())
    assert log["status"] == "failed"


def test_legacy_diagnosis_is_opt_in():
    legacy = dict(ATOMS, diagnosis=["Legacy fact"])
    assert set(normalize_atoms(legacy)) == set(ATOM_GROUPS)
    assert normalize_atoms(legacy, include_legacy_diagnosis=True)["diagnosis"] == ["Legacy fact"]
    with pytest.raises(ValueError, match="array of strings"):
        normalize_atoms(dict(ATOMS, diagnostics=[{"text": "wrong shape"}]))


def test_reference_keeps_source_text_in_separate_output(local_source, tmp_path):
    destination = tmp_path / "references" / "synthetic-case_gt.md"
    article_to_markdown(local_source / "article.xml", destination)
    markdown = destination.read_text()
    assert "ABSTRACT_ONLY_TEXT" in markdown
    assert "DISCUSSION_ONLY_TEXT" in markdown
    assert "CAPTION_ONLY_TEXT" in markdown
    assert "REFERENCE_ONLY_TEXT" in markdown
    assert "reference_images/synthetic-image.tif" in markdown
    assert (destination.parent / "reference_images" / "synthetic-image.tif").exists()


class FakeResponse:
    def __init__(self, *, data=None, content=b"asset", fail_stream=False):
        self.data, self.content, self.fail_stream = data, content, fail_stream

    def raise_for_status(self):
        pass

    def json(self):
        return self.data

    def iter_content(self, chunk_size):
        yield self.content
        if self.fail_stream:
            raise requests.ConnectionError("interrupted download")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeSession:
    def __init__(self, metadata, *, fail_stream=False):
        self.metadata, self.fail_stream, self.calls = metadata, fail_stream, []

    def get(self, url, **kwargs):
        self.calls.append(url)
        if "/metadata/" in url:
            return FakeResponse(data=self.metadata)
        return FakeResponse(fail_stream=self.fail_stream)


def test_downloader_uses_official_selected_id_and_resumes(tmp_path):
    metadata = {
        "xml_url": "s3://pmc-oa-opendata/PMC123.2/article.xml",
        "media_urls": ["s3://pmc-oa-opendata/PMC123.2/figure.png", "s3://pmc-oa-opendata/PMC123.2/movie.mp4"],
        "pdf_url": "s3://pmc-oa-opendata/PMC123.2/article.pdf",
    }
    session = FakeSession(metadata)
    result = download_article("123", tmp_path, session=session, version=2)
    assert result["files"] == ["article.xml", "figure.png"]
    assert session.calls[0] == BASE_URL + "/metadata/PMC123.2.json"
    session.calls.clear()
    download_article("PMC123", tmp_path, session=session, version=2)
    assert session.calls == [BASE_URL + "/metadata/PMC123.2.json"]
    assert normalize_pmcid("pmc123") == "PMC123"
    with pytest.raises(ValueError):
        normalize_pmcid("../../123")
    with pytest.raises(ValueError, match="official PMC"):
        collect_assets({"xml_url": "https://example.com/private.xml"})


def test_interrupted_download_does_not_leave_partial_final_file(tmp_path):
    session = FakeSession({"xml_url": "s3://pmc-oa-opendata/PMC123.1/article.xml"}, fail_stream=True)
    with pytest.raises(requests.ConnectionError):
        download_article("PMC123", tmp_path, session=session)
    assert not (tmp_path / "PMC123" / "article.xml").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_downloader_rejects_explicit_author_manuscript(tmp_path):
    session = FakeSession({"is_manuscript": True, "xml_url": "s3://pmc-oa-opendata/PMC123.1/article.xml"})
    with pytest.raises(ValueError, match="author manuscript"):
        download_article("PMC123", tmp_path, session=session)
    assert len(session.calls) == 1
