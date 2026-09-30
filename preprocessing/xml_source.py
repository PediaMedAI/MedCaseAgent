"""Small JATS reader shared by atom extraction and reference conversion.

Article identity is returned separately from clinical text. The extraction view
omits article front matter, bibliography, and non-clinical narrative sections.
This is structural filtering, not a substitute for reviewing extracted facts.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
EXCLUDED_SECTIONS = re.compile(
    r"\b(introduction|background|discussion|conclusions?|references|bibliography|"
    r"acknowledg\w*|funding|conflicts?|competing interests|author contributions|"
    r"ethics|consent|data availability|final diagnosis|diagnostic exposition)\b",
    re.IGNORECASE,
)
CASE_HEADING = re.compile(
    r"^cases?(?: (?:reports?|presentations?|descriptions?|summary|summaries|introduction))?(?: \d+)?$",
    re.IGNORECASE,
)
CLINICAL_HEADING = re.compile(
    r"^(?:patient information|patient history|clinical presentation|clinical findings|"
    r"clinical course|diagnostic assessment|diagnostic evaluation|therapeutic interventions?|"
    r"treatment|management|follow[ -]?up(?: (?:and|&) outcomes?)?|outcomes?)$",
    re.IGNORECASE,
)
ADMINISTRATIVE_SECTION = re.compile(
    r"^(?:ethics(?: statement| approval| declaration)?|ethical approval|"
    r"consent(?: statement| for publication)?|informed consent|patient(?:/parent)? consent)$",
    re.IGNORECASE,
)
ADMINISTRATIVE_PARAGRAPH = re.compile(
    r"\b(?:institutional review board|irb|review board|ethics?|ethical|consent|"
    r"declaration of helsinki|funding|author contributions?|conflicts? of interest|"
    r"competing interests|data availability|study (?:was |has been )?approved)\b",
    re.IGNORECASE,
)


def _section_heading(node: ET.Element) -> str:
    heading = element_text(node.find("title"))
    return re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", heading).strip(" .:").casefold()


def _case_section(node: ET.Element) -> bool:
    heading = _section_heading(node)
    # Case introduction is a clinical heading, but a cases attribute must not
    # override explicit discussion, diagnosis exposition, or administrative titles.
    other_exclusion = EXCLUDED_SECTIONS.search(re.sub(r"\bintroduction\b", "", heading))
    if other_exclusion:
        return False
    kind = node.get("sec-type", "").casefold().replace("_", "-")
    return kind in {"case", "cases", "case-report", "case-presentation"} or bool(CASE_HEADING.fullmatch(heading))


def _clinical_section(node: ET.Element) -> bool:
    return _case_section(node) or bool(CLINICAL_HEADING.fullmatch(_section_heading(node)))


def _excluded_section(node: ET.Element) -> bool:
    return bool(EXCLUDED_SECTIONS.search(f"{element_text(node.find('title'))} {node.get('sec-type', '')}"))


def _introduction_section(node: ET.Element) -> bool:
    return bool(re.search(r"\b(?:introduction|background)\b", _section_heading(node))) or node.get("sec-type", "").casefold() == "intro"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_article(path: str | Path) -> ET.Element:
    """Read a local JATS article without fetching external resources."""
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = local_name(node.tag)
    if root.tag == "article":
        return root
    article = root.find(".//article")
    if article is None:
        raise ValueError("XML contains no JATS article element")
    return article


def element_text(node: ET.Element | None, *, strip_links: bool = False, exclude_tags: set[str] | None = None) -> str:
    if node is None:
        return ""

    def pieces(element: ET.Element) -> list[str]:
        result = [element.text or ""]
        for child in element:
            if child.tag not in (exclude_tags or set()) and not (strip_links and child.tag in {"xref", "ext-link", "ref-list"}):
                result.extend(pieces(child))
            result.append(child.tail or "")
        return result

    return " ".join("".join(pieces(node)).split())


def source_identifiers(root: ET.Element) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    for node in root.findall("./front/article-meta/article-id"):
        kind = node.get("pub-id-type", "").lower()
        value = element_text(node)
        if kind in {"pmc", "pmcid"} and value:
            digits = re.sub(r"^PMC", "", value, flags=re.IGNORECASE)
            if digits.isdigit():
                identifiers["pmcid"] = f"PMC{digits}"
        elif kind == "pmid" and value.isdigit():
            identifiers["pmid"] = value
        elif kind == "doi" and value:
            identifiers["doi"] = re.sub(r"^(?:https?://(?:dx\.)?doi.org/|doi:\s*)", "", value, flags=re.IGNORECASE).strip().lower()
    return identifiers


def manuscript_sections(root: ET.Element) -> list[str]:
    """Return safe conventional headings in source order, without diagnostic titles.

    A free-form section title can itself disclose the source's diagnosis. Only
    familiar structural headings are exported as guidance for report writing.
    """
    known = {
        "abstract": "Abstract", "introduction": "Introduction",
        "background": "Background", "case": "Case Presentation",
        "case report": "Case Report", "case presentation": "Case Presentation",
        "case description": "Case Description", "clinical presentation": "Clinical Presentation",
        "discussion": "Discussion", "conclusion": "Conclusion",
        "conclusions": "Conclusions", "references": "References",
    }
    result = ["Abstract"] if root.find("./front/article-meta/abstract") is not None else []
    for node in root.findall("./body/sec"):
        title = element_text(node.find("title")).casefold().strip(" .:")
        title = re.sub(r"^\d+[.)]?\s*", "", title)
        if title in known and known[title] not in result:
            result.append(known[title])
    if root.find("./back/ref-list") is not None and "References" not in result:
        result.append("References")
    return result


def clinical_text(root: ET.Element, included_sections: list[str] | None = None) -> tuple[str, list[str]]:
    """Return body facts for compression; never include title/abstract/references.

    Legacy options ``authors``, ``year``, and ``citations`` are intentionally
    ignored. Figure/table captions, when requested, are extraction context only.
    Explicit clinical subsections may be recovered from introduction wrappers.
    An empty case container can recover non-administrative paragraphs misplaced
    under an ethics/consent heading; this never enables other excluded sections.
    External tables are included only when retained clinical content cites their
    IDs. Embedded figures and tables are handled separately from paragraph text.
    """
    optional = set(included_sections if included_sections is not None else ["tables"])
    parts: list[str] = []
    headers: list[str] = []
    case_containers: list[ET.Element] = []
    paragraph_exclusions = {"fig", "table-wrap", "ref-list", "back", "fn-group", "supplementary-material"}
    referenced_table_ids: list[str] = []
    rendered_tables: set[int] = set()
    external_tables = {
        table.get("id"): table
        for group in root.findall("floats-group")
        for table in group.findall(".//table-wrap")
        if table.get("id")
    }
    body = root.find("body")
    if body is None:
        raise ValueError("Article has no body; an abstract alone is not a case source")

    def record_table_reference(node: ET.Element) -> None:
        if "tables" in optional and node.get("ref-type") == "table":
            for identifier in node.get("rid", "").split():
                if identifier not in referenced_table_ids:
                    referenced_table_ids.append(identifier)

    def collect_paragraph_table_references(node: ET.Element) -> None:
        if node.tag in paragraph_exclusions:
            return
        if node.tag == "xref":
            record_table_reference(node)
            return
        for child in node:
            collect_paragraph_table_references(child)

    def visit_embedded_assets(node: ET.Element) -> None:
        for child in node:
            if child.tag in {"fig", "table-wrap"}:
                visit(child)
            elif child.tag not in paragraph_exclusions:
                visit_embedded_assets(child)

    def append_referenced_tables() -> None:
        for identifier in referenced_table_ids:
            table = external_tables.get(identifier)
            if table is not None:
                visit(table)

    def visit_clinical_descendants(node: ET.Element) -> None:
        # A malformed JATS hierarchy sometimes nests the case below Introduction.
        # Inspect section structure only: the wrapper's own paragraphs stay out.
        for child in node:
            if child.tag != "sec":
                continue
            if _clinical_section(child):
                visit(child)
            elif not _excluded_section(child) or _introduction_section(child):
                visit_clinical_descendants(child)

    def visit(node: ET.Element) -> None:
        if node.tag == "sec":
            title = element_text(node.find("title"))
            if _excluded_section(node) and not _clinical_section(node):
                if _introduction_section(node):
                    visit_clinical_descendants(node)
                return
            if _case_section(node):
                case_containers.append(node)
            if title:
                headers.append(title)
        if node.tag in {"ref-list", "back", "fn-group", "supplementary-material"}:
            return
        if node.tag == "xref":
            record_table_reference(node)
            return
        if node.tag == "fig":
            if "figures" in optional:
                caption = element_text(node.find("caption"), strip_links=True)
                if caption:
                    parts.append("Image finding context: " + caption)
            return
        if node.tag == "table-wrap":
            if "tables" in optional and id(node) not in rendered_tables:
                rendered_tables.add(id(node))
                caption = element_text(node.find("caption"), strip_links=True)
                rows = [" | ".join(element_text(cell, strip_links=True) for cell in row if cell.tag in {"td", "th"}) for row in node.findall(".//tr")]
                table = "\n".join([caption, *rows]).strip()
                if table:
                    parts.append("Clinical table:\n" + table)
            return
        if node.tag == "p":
            text = element_text(node, strip_links=True, exclude_tags=paragraph_exclusions)
            if "Springer Nature remains neutral" not in text and "Publisher's Note" not in text:
                if text:
                    parts.append(text)
                collect_paragraph_table_references(node)
                visit_embedded_assets(node)
            return
        for child in node:
            if child.tag not in {"title", "label"}:
                visit(child)

    visit(body)
    append_referenced_tables()
    if not parts and case_containers:
        # Some sources place the entire case narrative in an Ethics subsection.
        # Recovery is limited to such administrative subsections of a recognized
        # case container, after the standard view has produced no clinical text.
        seen_paragraphs: set[int] = set()

        def recover_case(node: ET.Element, in_administrative_section: bool = False) -> None:
            if node.tag == "sec":
                administrative = bool(ADMINISTRATIVE_SECTION.fullmatch(_section_heading(node)))
                if _excluded_section(node) and not _clinical_section(node) and not administrative:
                    return
                in_administrative_section = in_administrative_section or administrative
            if node.tag in {"ref-list", "back", "fn-group", "supplementary-material", "fig", "table-wrap"}:
                return
            if node.tag == "p":
                if in_administrative_section and id(node) not in seen_paragraphs:
                    seen_paragraphs.add(id(node))
                    text = element_text(node, strip_links=True, exclude_tags=paragraph_exclusions)
                    if not ADMINISTRATIVE_PARAGRAPH.search(text) and "Springer Nature remains neutral" not in text and "Publisher's Note" not in text:
                        if text:
                            parts.append(text)
                        collect_paragraph_table_references(node)
                return
            for child in node:
                if child.tag not in {"title", "label"}:
                    recover_case(child, in_administrative_section)

        for container in case_containers:
            recover_case(container)
        append_referenced_tables()
    if not parts:
        raise ValueError("No clinical body paragraphs remain after section filtering")
    return "\n\n".join(parts), list(dict.fromkeys(headers))


def find_xml_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".xml", ".nxml"})


def find_images(directory: Path) -> list[Path]:
    """Find local image assets, including common ``imgs``/``images`` layouts."""
    candidates = list(directory.iterdir())
    for name in ("imgs", "images", "media"):
        child = directory / name
        if child.is_dir():
            candidates.extend(child.iterdir())
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
