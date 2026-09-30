"""Convert source JATS articles to reference Markdown for separate inspection.

This replaces the research ``convertion.py`` Markdown path without its mandatory
PDF/font dependencies. Reference reports are never an input to atom generation
or the report-writing agent.
"""

from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from preprocessing.xml_source import element_text, find_images, find_xml_files, local_name, parse_article


def _attribute(node: ET.Element, name: str) -> str:
    return next((value for key, value in node.attrib.items() if local_name(key) == name), "")


def article_to_markdown(
    xml_path: str | Path,
    output_path: str | Path,
    *,
    display_authors: bool = False,
    append_unmatched_figures: bool = True,
) -> Path:
    """Write an article's reference text and referenced images to a separate folder."""
    xml_path, output_path = Path(xml_path), Path(output_path)
    root = parse_article(xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    images = find_images(xml_path.parent)
    rendered_figures: set[int] = set()

    def render_figure(node: ET.Element) -> None:
        if id(node) in rendered_figures:
            return
        rendered_figures.add(id(node))
        label = element_text(node.find("label")) or "Figure"
        caption = element_text(node.find("caption"))
        for graphic in node.findall(".//graphic"):
            href = _attribute(graphic, "href")
            name = Path(href).name
            candidate = next((path for path in images if path.name == name), None)
            if candidate is None:
                candidate = next((path for path in images if path.stem == Path(name).stem), None)
            if candidate is None:
                lines.append(f"<!-- Missing local image asset for {label}. -->")
                continue
            image_dir = output_path.parent / "reference_images"
            image_dir.mkdir(exist_ok=True)
            target = image_dir / candidate.name
            if candidate.resolve() != target.resolve():
                shutil.copyfile(candidate, target)
            alt = label.replace("[", "").replace("]", "")
            lines.append(f"![{alt}](reference_images/{candidate.name})")
        if caption:
            lines.append(f"**{label}.** {caption}")

    def render(node: ET.Element, depth: int = 2) -> None:
        if node.tag == "sec":
            title = element_text(node.find("title"))
            if title:
                lines.append(f"{'#' * min(depth, 6)} {title}")
            for child in node:
                if child.tag != "title":
                    render(child, depth + 1)
        elif node.tag == "p":
            text = element_text(node)
            if text:
                lines.append(text)
        elif node.tag == "fig":
            render_figure(node)
        elif node.tag == "table-wrap":
            label = element_text(node.find("label"))
            caption = element_text(node.find("caption"))
            if label or caption:
                lines.append(f"**{label or 'Table'}.** {caption}")
            rows = [[element_text(cell).replace("|", "\\|") for cell in row if cell.tag in {"td", "th"}] for row in node.findall(".//tr")]
            rows = [row for row in rows if row]
            if rows:
                width = max(map(len, rows))
                rows = [row + [""] * (width - len(row)) for row in rows]
                table_lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
                table_lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
                lines.append("\n".join(table_lines))
        elif node.tag not in {"ref-list", "title", "label"}:
            for child in node:
                render(child, depth)

    title = element_text(root.find("./front/article-meta/title-group/article-title"))
    lines.append(f"# {title or 'Source case report'}")
    if display_authors:
        authors = [element_text(node.find("name")) for node in root.findall("./front/article-meta/contrib-group/contrib") if node.get("contrib-type") == "author"]
        if any(authors):
            lines.append("; ".join(filter(None, authors)))
    abstract = root.find("./front/article-meta/abstract")
    if abstract is not None:
        lines.append("## Abstract")
        render(abstract)
    body = root.find("body")
    if body is not None:
        render(body)
    if append_unmatched_figures:
        remaining = [node for node in root.findall(".//fig") if id(node) not in rendered_figures]
        if remaining:
            lines.append("## Additional figures")
            for figure in remaining:
                render_figure(figure)
    references = root.findall("./back/ref-list/ref")
    if references:
        lines.append("## References")
        for index, reference in enumerate(references, 1):
            citation = next((child for child in reference if child.tag in {"element-citation", "mixed-citation"}), reference)
            # JATS element-citation often uses adjacent tags without whitespace.
            text = " ".join(filter(None, (element_text(child) for child in citation))) or element_text(citation)
            text = re.sub(r"\s+", " ", text)
            lines.append(f"{index}. {text}")
    output_path.write_text("\n\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


class MDConversionPipeline:
    """Compatibility wrapper for the research Markdown conversion entry point."""

    def __init__(
        self,
        data_dir: str | Path,
        out_dir: str | Path,
        display_authors: bool = False,
        generated_style: bool = True,
        append_unmatched_figures: bool = True,
        default_section_title: str | None = None,
        doc_ids: list[str] | None = None,
    ):
        self.data_dir, self.out_dir = Path(data_dir), Path(out_dir)
        if self.data_dir.resolve() == self.out_dir.resolve():
            raise ValueError("Source and reference output directories must differ")
        self.display_authors = display_authors
        self.append_unmatched_figures = append_unmatched_figures
        self.doc_ids = doc_ids
        # Kept as accepted arguments for callers of the original constructor.
        self.generated_style = generated_style
        self.default_section_title = default_section_title

    def run(self) -> dict[str, int]:
        if not self.data_dir.is_dir():
            raise ValueError(f"Source directory does not exist: {self.data_dir}")
        folders = sorted(path for path in self.data_dir.iterdir() if path.is_dir() and find_xml_files(path))
        if self.doc_ids is not None:
            folders = [path for path in folders if path.name in self.doc_ids]
        if not folders:
            raise ValueError("No selected case subdirectories contain XML/NXML")
        success = 0
        for folder in folders:
            errors = []
            for xml_path in find_xml_files(folder):
                try:
                    article_to_markdown(xml_path, self.out_dir / folder.name / f"{folder.name}_gt.md", display_authors=self.display_authors, append_unmatched_figures=self.append_unmatched_figures)
                    success += 1
                    break
                except (OSError, ValueError, SyntaxError) as error:
                    errors.append(str(error))
            else:
                print(f"{folder.name}: reference conversion failed: {'; '.join(errors)}")
        return {"total": len(folders), "success": success, "failed": len(folders) - success}
