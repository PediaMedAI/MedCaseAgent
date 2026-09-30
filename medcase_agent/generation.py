import os
import re
import json
import glob
import shutil
import base64
import time
import uuid
from datetime import datetime
from typing import Any, Iterable
from pathlib import Path

from .utils import get_openai_client, finalize_prompt, generate_llm_response, format_clinical_section
from .tools.registry import TOOL_SCHEMAS, AVAILABLE_TOOLS
from .tools.disease_importance_tools import _normalize_identifier

class GenerationPipeline:
    MIN_VERIFIED_REFERENCES = 10
    TARGET_REFERENCE_COUNT = 12
    MAX_REFERENCE_COUNT = 15

    def __init__(self, working_dir: str, model_id: str = None, mode: str = "multi", client=None, tools_config: dict = None):
        """
        Initializes the pipeline to generate medical journal articles from JSON atoms.
        :param mode: "single" for one-shot generation, "multi" for the multi-agent staged pipeline.
        """
        self.working_dir = working_dir
        self.model_id = model_id or os.getenv("OPENAI_MODEL")
        if not self.model_id:
            raise ValueError("Set OPENAI_MODEL or supply model_id")
        self.mode = mode.lower()
        if self.mode not in {"single", "multi"}:
            raise ValueError("mode must be single or multi")
        self.client = client if client else get_openai_client()
        self.tools_config = tools_config if isinstance(tools_config, dict) else {}

        tools_block = self._tools_block()
        multi_agent_cfg = tools_block.get("multi_agent", self.tools_config.get("multi_agent", {}))
        if not isinstance(multi_agent_cfg, dict):
            multi_agent_cfg = {}
        # Each pipeline phase already starts from a fresh message history. These flags
        # make that explicit in the trajectory log and optionally create a fresh API
        # client per agent/session when requested by YAML config.
        self.multi_agent_enabled = bool(multi_agent_cfg.get("enabled", True))
        self.fresh_client_per_agent = bool(multi_agent_cfg.get("fresh_client_per_agent", False))

    def _tools_block(self) -> dict:
        """Returns the nested tools config block while supporting flat configs."""
        if not isinstance(self.tools_config, dict):
            return {}
        block = self.tools_config.get("tools", self.tools_config)
        return block if isinstance(block, dict) else {}

    def _retrieval_exclusions(self, case_data: dict) -> dict:
        """Collect provenance identifiers without exposing source prose to a model."""
        result = {"exclude_pmcids": set(), "exclude_pmids": set(), "exclude_dois": set()}

        def visit(value):
            if isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for name, item in value.items():
                    key = str(name).lower().replace("_", "")
                    kind = next((kind for kind in ("pmcid", "pmid", "doi") if kind in key), None)
                    if kind and not isinstance(item, dict):
                        for identifier in item if isinstance(item, (list, tuple, set)) else [item]:
                            if identifier is None or isinstance(identifier, (dict, list)):
                                continue
                            text = _normalize_identifier(identifier, kind)
                            if text:
                                result[f"exclude_{kind}s"].add(text)
                    elif isinstance(item, (dict, list)):
                        visit(item)

        for value in (
            case_data.get("metadata", {}), case_data.get("_provenance", {}),
            case_data.get("source_identifiers", {}), case_data.get("related_exclusions", {}),
            case_data.get("exclusions", {}), case_data.get("_retrieval_exclusions", {}),
            self.tools_config.get("exclusions", {}),
        ):
            visit(value)
        return {key: sorted(values) for key, values in result.items()}

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _copy_images(self, source_directory: str, destination_directory: str):
        if not os.path.exists(source_directory):
            print(f"[-] Source directory not found for images: {source_directory}")
            return 0
        copied_count = 0
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            for img_path in glob.glob(os.path.join(source_directory, ext)):
                try:
                    shutil.copy2(img_path, destination_directory)
                    copied_count += 1
                except Exception as e:
                    print(f"[!] Warning: Failed to copy image {img_path}: {e}")
        return copied_count

    def _normalize_section_name(self, section_name: str) -> str:
        """Normalizes a section label so prompts can compare headings robustly."""
        return re.sub(r"[^a-z0-9]+", "", str(section_name or "").strip().lower())


    def _section_names_from_str(self, sections_str: str) -> list:
        """Extracts clean section names from a Markdown bullet-list string."""
        names = []
        for raw_line in str(sections_str or "").splitlines():
            name = raw_line.strip().lstrip("-").strip()
            if name:
                names.append(name)
        return names

    def _has_section(self, sections: Iterable[str], section_name: str) -> bool:
        target = self._normalize_section_name(section_name)
        return any(self._normalize_section_name(item) == target for item in sections or [])

    def _add_unique_section(self, sections: list, section_name: str):
        name = str(section_name or "").strip().strip("-").strip()
        if not name:
            return
        norm = self._normalize_section_name(name)
        if not norm:
            return
        if norm not in {self._normalize_section_name(x) for x in sections}:
            sections.append(name)

    def _front_matter_section_names(self, sections_str: str) -> list:
        """Returns the source-driven front matter sections that should be generated in phase 3."""
        names = self._section_names_from_str(sections_str)
        front_norms = {"title", "keywords", "abstract"}
        front = [name for name in names if self._normalize_section_name(name) in front_norms]
        # Title is mandatory for publication and is explicitly required in phase 3.
        if not self._has_section(front, "Title"):
            front.insert(0, "Title")
        return front

    def _section_bullets(self, sections: Iterable[str]) -> str:
        return "\n".join(f"- {section}" for section in sections if str(section or "").strip())

    def _schemas_for_tools(self, tool_names: Iterable[str]) -> list:
        """Selects tool schemas by function name."""
        wanted = set(tool_names or [])
        return [
            schema for schema in TOOL_SCHEMAS
            if schema.get("function", {}).get("name") in wanted
        ]

    def _format_tool_result_for_prompt(self, raw_result: Any) -> str:
        """Stringifies tool output without losing structured citation data."""
        if raw_result is None:
            return ""
        if isinstance(raw_result, str):
            return raw_result
        try:
            return json.dumps(raw_result, ensure_ascii=False, indent=2)
        except TypeError:
            return str(raw_result)

    def _flatten_citation_items(self, raw_result: Any) -> list:
        """Best-effort extraction of formatted citation strings from citation-tool output."""
        if raw_result is None:
            return []
        if isinstance(raw_result, str):
            text = raw_result.strip()
            if not text or text.lower().startswith(("error", "no verified citations")):
                return []
            try:
                parsed = json.loads(text)
                return self._flatten_citation_items(parsed)
            except Exception:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                citation_like = [
                    re.sub(r"^\s*(?:\d+[\.)]|\[\d+\])\s*", "", line).strip()
                    for line in lines
                    if re.search(r"\b(doi|pmid|[12][0-9]{3})\b", line, flags=re.IGNORECASE)
                ]
                return citation_like or [text]
        if isinstance(raw_result, list):
            items = []
            for item in raw_result:
                items.extend(self._flatten_citation_items(item))
            return items
        if isinstance(raw_result, dict):
            for key in (
                "citations", "ama_citations", "formatted_citations", "references",
                "formatted", "results", "data", "items"
            ):
                value = raw_result.get(key)
                if value:
                    items = self._flatten_citation_items(value)
                    if items:
                        return items
            # Some tools return {doi: citation}; keep string-like values.
            items = []
            for value in raw_result.values():
                if isinstance(value, (str, list, dict)):
                    items.extend(self._flatten_citation_items(value))
            return items
        return [str(raw_result)]

    def _collect_verified_citations_from_logs(self, *logs: dict) -> list:
        """Collects de-duplicated citations that came from fetch_ama_citations tool calls."""
        citations = []
        seen = set()
        for log in logs:
            if not isinstance(log, dict):
                continue
            for turn in log.get("turns", []):
                for tc in turn.get("tool_calls", []):
                    if tc.get("tool_name") != "fetch_ama_citations":
                        continue
                    for item in self._flatten_citation_items(tc.get("raw_result")):
                        clean = re.sub(r"\s+", " ", str(item or "").strip())
                        clean = re.sub(r"^\s*(?:\d+[\.)]|\[\d+\])\s*", "", clean).strip()
                        if not clean:
                            continue
                        norm = clean.lower()
                        if norm not in seen:
                            citations.append(clean)
                            seen.add(norm)
        return citations

    def _citation_bank_for_prompt(self, citations: list) -> str:
        if not citations:
            return (
                "No verified citation-tool bank was captured yet. If the manuscript requires citations, "
                "use search_pubmed and fetch_ama_citations before adding any new reference strings."
            )
        lines = [
            f"{idx}. {citation}"
            for idx, citation in enumerate(citations[: self.MAX_REFERENCE_COUNT], start=1)
        ]
        return "\n".join(lines)

    def _reference_section_text(self, markdown: str) -> str:
        match = re.search(
            r"(?ims)^\s*#{1,6}\s*References\s*$([\s\S]*?)(?=^\s*#{1,6}\s+\S|\Z)",
            markdown or "",
        )
        if match:
            return match.group(1)
        # Fallback for unheaded plain text references near the end.
        match = re.search(r"(?ims)^\s*References\s*$([\s\S]*?)\Z", markdown or "")
        return match.group(1) if match else ""

    def _count_reference_items(self, markdown: str) -> int:
        refs = self._reference_section_text(markdown)
        if not refs.strip():
            return 0
        numbered = re.findall(r"(?m)^\s*(?:\d+[\.)]|\[\d+\])\s+", refs)
        if numbered:
            return len(numbered)
        # Fallback: count citation-like non-empty lines.
        lines = [line for line in refs.splitlines() if re.search(r"\b(doi|pmid|[12][0-9]{3})\b", line, re.I)]
        return len(lines)

    def _expand_citation_token(self, token: str) -> set:
        nums = set()
        for part in re.split(r",", token or ""):
            part = part.strip()
            if not part:
                continue
            range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if start <= end and end - start <= 50:
                    nums.update(range(start, end + 1))
            elif part.isdigit():
                nums.add(int(part))
        return nums

    def _count_inline_citations(self, markdown: str) -> int:
        # Remove the References section before counting inline citation markers.
        body = re.sub(
            r"(?ims)^\s*#{1,6}\s*References\s*$[\s\S]*$",
            "",
            markdown or "",
        )
        found = set()
        for token in re.findall(r"\[((?:\d+\s*(?:[-–,]\s*)?)+)\]", body):
            found.update(self._expand_citation_token(token))
        return len(found)

    def _citation_quality_report(self, markdown: str) -> dict:
        return {
            "minimum_required_references": self.MIN_VERIFIED_REFERENCES,
            "reference_items": self._count_reference_items(markdown),
            "unique_inline_citations": self._count_inline_citations(markdown),
        }

    def _needs_citation_repair(self, markdown: str) -> bool:
        report = self._citation_quality_report(markdown)
        return (
            report["reference_items"] < self.MIN_VERIFIED_REFERENCES
            or report["unique_inline_citations"] < self.MIN_VERIFIED_REFERENCES
        )
    def _build_care_sections_str(self, paper_sections: list, optional_sections: list = None) -> str:
        """
        Builds the manuscript section list from atom JSON metadata.

        The atom files already carry `metadata.paper_sections_found`; this method now
        preserves that source order instead of expanding every case into a fixed CARE
        heading template. CARE requirements are still enforced by folding CARE content
        into the closest source heading, especially "Case Report", when granular CARE
        headings are absent. Title and References are added if missing because they are
        required for publication and citation integrity.
        """
        sections = []
        for section in list(paper_sections or []) + list(optional_sections or []):
            self._add_unique_section(sections, section)

        if not sections:
            # Conservative fallback only when the atoms do not expose source headings.
            sections = ["Abstract", "Introduction", "Case Report", "Discussion", "Conflicts of Interest"]

        if not self._has_section(sections, "Title"):
            sections.insert(0, "Title")

        if not any(
            self._normalize_section_name(x) in {"references", "citations"}
            for x in sections
        ):
            sections.append("References")

        return self._section_bullets(sections)

    def _section_lines_without(self, sections_str: str, excluded_names: set) -> str:
        """Returns a Markdown bullet list of section headings, excluding given names."""
        excluded_norms = {self._normalize_section_name(x) for x in excluded_names}
        lines = []
        for raw_line in str(sections_str or "").splitlines():
            name = raw_line.strip().lstrip("-").strip()
            if not name or self._normalize_section_name(name) in excluded_norms:
                continue
            lines.append(f"- {name}")
        return "\n".join(lines)

    def _clinical_atoms_for_prompt(self, case_data: dict) -> dict:
        """Formats raw clinical atoms consistently across all generation phases."""
        return {
            "history": format_clinical_section(case_data.get("history")),
            "presentation": format_clinical_section(case_data.get("presentation")),
            "diagnostics": format_clinical_section(case_data.get("diagnostics")),
            "management": format_clinical_section(case_data.get("management")),
            "outcome": format_clinical_section(case_data.get("outcome")),
        }

    def _care_guidance(self) -> str:
        return """CARE REPORTING REQUIREMENTS:
- Preserve the exact section headings and order supplied in the REQUIRED SECTIONS list, which is derived from the atom JSON metadata.
- Do not invent missing section headings merely to satisfy the CARE checklist. When source sections are broad, fold CARE items into the closest available heading: patient information, clinical findings, timeline, diagnostics, interventions, and outcomes usually belong inside Case Report.
- Title: include the main diagnosis or intervention and the words "case report".
- Abstract: no citations; include why the case is notable, the patient's main concerns and important findings, primary diagnoses/interventions/outcomes, and a concise take-away lesson, should be short and concise.
- Introduction: provide focused background and clinical relevance with verified citations.
- Case Report or equivalent clinical section: include de-identified demographics, relevant medical/family/psychosocial/genetic history, main concerns, clinical findings, timeline, diagnostics, differential diagnosis, interventions, follow-up, outcomes, adherence, tolerability, and unresolved issues when present in the source atoms.
- Timeline: when the metadata contains a Timeline section, present sequence of events clearly; otherwise preserve real intervals and missed visits within the Case Report rather than smoothing them away.
- Diagnostic Assessment and Therapeutic Intervention: when these headings are absent, include the same substance inside Case Report using clear paragraphs.
- Discussion: include strengths and limitations of this case, relevant literature context, clinical reasoning, comparison with prior literature, and take-away lessons.
- Patient Perspective, Informed Consent, Ethics, or Conflicts of Interest: do not fabricate these. If the source atoms do not contain them, state that the information was not available in the source material or that no conflicts were reported, as appropriate to the provided atoms."""

    def _citation_style_guidance(self) -> str:
        return f"""CITATION STYLE AND DENSITY REQUIREMENTS:
- Use only references retrieved through the citation tools; do not invent DOIs, PMIDs, authors, years, journals, or reference strings.
- The final manuscript MUST contain no fewer than {self.MIN_VERIFIED_REFERENCES} unique verified references and no fewer than {self.MIN_VERIFIED_REFERENCES} distinct in-text citation numbers. Target {self.TARGET_REFERENCE_COUNT} high-value references and avoid exceeding {self.MAX_REFERENCE_COUNT} unless clinically necessary.
- Stage 2 drafting must already include inline citations and a References section when References is listed in the required sections; do not wait until the editor stage to add all citations.
- Do not cite raw patient facts from the provided atoms; cite only background, epidemiology, guidelines, diagnostic criteria, treatment rationale, or comparison with prior literature.
- Avoid citation stuffing. Use no more than one citation marker at the end of a sentence in routine background statements. Use two citations in one sentence only when the sentence explicitly compares or synthesizes two sources. Never place three or more citation markers in a single sentence.
- Distribute citations naturally across the Introduction and Discussion, with occasional citations in Diagnostic Assessment, Therapeutic Intervention, or Case Report only when literature support is needed.
- Do not place citations in the Abstract, figure captions, Timeline, Patient Perspective, Informed Consent, or Conflicts of Interest.
- Every item in References must be cited at least once, and every inline citation must have a matching References entry."""

    def _figure_integration_guidance(self, num_images, ids_string: str) -> str:
        return f"""FIGURE INTEGRATION REQUIREMENTS:
- There are exactly {num_images} image(s). Reference IDs: [{ids_string}].
- Embed every source image exactly once using Markdown syntax: ![Figure n](IMG_XXXXXX), followed immediately by a compact caption line beginning with > **Figure n:**.
- Figures may contain A/B or other labeled panels. Mention clinically relevant panels in the main text as Figure 1A, Figure 1B, etc.
- Each figure must be referenced and clinically interpreted in the main text before the image appears. The main-text sentence must explicitly name the figure(Figure 1A and 1B). Do not leave the image explanation only in the caption. Place each figure near the relevant source section.
- Captions must be compact and concise, preferably one sentence and no more than 50 words. For A/B panels, use a concise format such as "(A) finding; (B) finding." Do not include citations in captions."""

    def _extract_thinking(self, content: str) -> tuple:
        """Extracts <think> tags from reasoning models to save in the trajectory."""
        if not content:
            return None, content

        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_process = think_match.group(1).strip()
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            return thinking_process, clean_content
        return None, content

    def _prepare_multimodal_payload(self, text_prompt: str, source_dir: str) -> tuple:
        all_imgs = []
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            all_imgs.extend(glob.glob(os.path.join(source_dir, ext)))
        all_imgs.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])

        image_map = {}
        virtual_ids_list = []

        for img_p in all_imgs:
            virtual_id = f"IMG_{uuid.uuid4().hex[:6].upper()}"
            image_map[virtual_id] = img_p
            virtual_ids_list.append(virtual_id)

        ids_string = ", ".join(virtual_ids_list) if virtual_ids_list else "None"
        final_prompt = text_prompt.replace("{IMAGE_IDS}", ids_string)
        final_prompt = final_prompt.replace("{NUM_IMAGES}", str(len(virtual_ids_list)))

        content_payload = [{"type": "text", "text": final_prompt}]
        for virtual_id, img_p in image_map.items():
            b64 = self._encode_image(img_p)
            mime = "jpeg" if img_p.lower().endswith(('.jpg', '.jpeg')) else "png"
            content_payload.append({"type": "text", "text": f"Reference ID: {virtual_id}"})
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{b64}", "detail": "high"}
            })
            image_map[virtual_id] = os.path.basename(img_p)

        return content_payload, image_map, virtual_ids_list

    def _execute_llm_loop(self, messages: list, allowed_tools: list, case_data: dict, phase_name: str, image_map: dict = None, agent_name: str = None, max_turns: int = 12) -> dict:
        """Generic loop for handling LLM generation, tool calls, and thought extraction."""
        phase_start_time = datetime.now()
        agent_name = agent_name or phase_name
        session_id = f"{agent_name}_{uuid.uuid4().hex[:10]}" if self.multi_agent_enabled else None
        agent_client = get_openai_client() if (self.multi_agent_enabled and self.fresh_client_per_agent) else self.client

        # 1. Dynamically extract the system prompt from the messages payload
        system_prompt = next((msg["content"] for msg in messages if msg.get("role") == "system"), None)

        # 2. Extract the user prompt, ignoring base64 image data
        user_prompt_text = None
        user_msg = next((msg["content"] for msg in messages if msg.get("role") == "user"), None)

        if isinstance(user_msg, list):
            text_parts = [item["text"] for item in user_msg if item.get("type") == "text"]
            user_prompt_text = "\n".join(text_parts)
        elif isinstance(user_msg, str):
            user_prompt_text = user_msg

        # 3. Extract the image paths/filenames used in this phase
        images_used = list(image_map.values()) if image_map else []

        phase_log = {
            "phase": phase_name,
            "agent_name": agent_name,
            "agent_session_id": session_id,
            "fresh_client_per_agent": bool(self.multi_agent_enabled and self.fresh_client_per_agent),
            "start_time": phase_start_time.isoformat(),
            "end_time": None,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt_text,
            "images_used": images_used,
            "api_call_count": 0,
            "turns": [],
            "final_output": None,
            "total_prompt_tokens": 0,
            "total_comp_tokens": 0,
            "mapped_images": image_map or {},
            "max_turns": max_turns,
        }

        print(f"\n    === Starting {phase_name} | Agent: {agent_name} | Session: {session_id or 'shared'} ===")

        for turn in range(max_turns):
            phase_log["api_call_count"] += 1
            print(f"    [*] Turn {turn + 1}: Generating... ", end="", flush=True)

            response_data = generate_llm_response(
                client=agent_client,
                model=self.model_id,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                tools=allowed_tools if allowed_tools else None,
                tool_choice="auto" if allowed_tools else None,
                temperature=0.1
            )
            print()

            if response_data.get("usage"):
                usage = response_data["usage"]
                phase_log["total_prompt_tokens"] += usage.get("prompt_tokens", 0) if isinstance(usage, dict) else (usage.prompt_tokens or 0)
                phase_log["total_comp_tokens"] += usage.get("completion_tokens", 0) if isinstance(usage, dict) else (usage.completion_tokens or 0)

            raw_content = response_data.get("content", "")
            reasoning_content = response_data.get("reasoning_content", "")
            tool_calls = response_data.get("tool_calls", [])

            extracted_thinking, clean_content = self._extract_thinking(raw_content)
            final_thinking = reasoning_content if reasoning_content else extracted_thinking

            turn_log = {
                "turn": turn + 1,
                "thinking": None,
                "content": clean_content,
                "tool_calls": []
            }

            if not tool_calls:
                print(f"    [*] {phase_name} complete.")
                if not clean_content or not clean_content.strip():
                    raise RuntimeError(f"{phase_name} returned an empty response")
                phase_log["final_output"] = clean_content
                phase_log["turns"].append(turn_log)
                phase_log["end_time"] = datetime.now().isoformat()
                return phase_log

            messages.append({
                "role": "assistant",
                "content": raw_content if raw_content else None,
                "tool_calls": tool_calls
            })

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args_str = tool_call["function"]["arguments"]
                allowed_names = {schema["function"]["name"] for schema in allowed_tools}
                function_to_call = AVAILABLE_TOOLS.get(function_name) if function_name in allowed_names else None

                tool_exec_log = {"tool_name": function_name, "arguments": None, "raw_result": None, "error": None}

                try:
                    function_args = json.loads(function_args_str)
                    if not isinstance(function_args, dict):
                        raise ValueError("Tool arguments must be a JSON object")

                    # Runtime-only context cannot be overwritten by model arguments.
                    function_args["case_data"] = case_data
                    function_args["execution_log"] = phase_log
                    for key, values in self._retrieval_exclusions(case_data).items():
                        function_args[key] = values
                    if function_name == "analyze_radiology_image":
                        function_args["vllm_url"] = os.getenv("MEDCASE_MEDGEMMA_BASE_URL")
                        function_args["vllm_model"] = os.getenv("MEDCASE_MEDGEMMA_MODEL")
                        function_args["use_vllm"] = bool(function_args["vllm_url"])
                    elif function_name == "analyze_composite_figure":
                        function_args["model_id"] = os.getenv("MEDCASE_VISION_MODEL") or self.model_id
                        function_args["base_url"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                        function_args["api_key_env"] = "OPENAI_API_KEY"

                    # Keep internal kwargs out of the terminal print
                    excluded_keys = [
                        "execution_log", "case_data",
                        "use_vllm", "vllm_url", "vllm_model",
                        "model_id", "base_url", "api_key_env"
                    ]
                    log_args = {k: v for k, v in function_args.items() if k not in excluded_keys}
                    tool_exec_log["arguments"] = log_args
                    print(f"        -> Tool: {function_name}({log_args})")

                    if function_to_call:
                        function_response = function_to_call(**function_args)
                        try:
                            tool_exec_log["raw_result"] = json.loads(function_response) if isinstance(function_response, str) else function_response
                        except json.JSONDecodeError:
                            tool_exec_log["raw_result"] = function_response
                    else:
                        function_response = f"Error: Tool '{function_name}' is not registered."
                        tool_exec_log["error"] = function_response

                except Exception as e:
                    print(f"        [X] Tool execution failed: {e}")
                    function_response = f"Error executing tool: {str(e)}"
                    tool_exec_log["error"] = str(e)

                turn_log["tool_calls"].append(tool_exec_log)
                messages.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": function_response if isinstance(function_response, str) else json.dumps(function_response, ensure_ascii=False),
                })

            phase_log["turns"].append(turn_log)

        phase_log["end_time"] = datetime.now().isoformat()
        raise RuntimeError(f"{phase_name} exceeded {max_turns} tool turns without a final response")

    def _post_process_markdown(self, raw_markdown: str, image_map: dict) -> str:
        """Replaces virtual image IDs with real local paths."""
        processed = raw_markdown.strip()
        for v_id, real_name in image_map.items():
            processed = re.sub(rf"\(({re.escape(v_id)})\)", f"(imgs/{real_name})", processed)
        return processed.strip()

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
        return slug or "phase"

    def _code_fence(self, text: str, language: str = "") -> str:
        raw = "" if text is None else str(text)
        longest = max((len(match.group(0)) for match in re.finditer(r"`+", raw)), default=0)
        fence = "`" * max(3, longest + 1)
        header = f"{fence}{language}".rstrip()
        return f"{header}\n{raw}\n{fence}"

    def _format_value_for_markdown(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

    def _value_code_block(self, value: Any, default_language: str = "text") -> str:
        language = "json" if isinstance(value, (dict, list)) else default_language
        return self._code_fence(self._format_value_for_markdown(value), language)

    def _phase_entries_from_result(self, result_data: dict) -> list:
        phases_block = result_data.get("phases", {})
        if isinstance(phases_block, dict) and phases_block:
            return [
                (phase_key, phase_log)
                for phase_key, phase_log in phases_block.items()
                if isinstance(phase_log, dict)
            ]

        fallback_entries = []
        for idx, key in enumerate(("execution", "audit", "citation_repair")):
            phase_log = result_data.get(key)
            if isinstance(phase_log, dict):
                fallback_entries.append((f"phase_{idx}", phase_log))
        return fallback_entries

    def _phase_sequence_number(self, phase_key: str, phase_log: dict) -> int:
        for candidate in (phase_key, phase_log.get("phase")):
            match = re.search(r"phase[_\s-]*(\d+)", str(candidate or ""), flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 999

    def _collect_visual_tool_outputs_from_logs(self, *logs: dict) -> list:
        visual_tools = {"analyze_composite_figure", "analyze_radiology_image"}
        items = []
        for log in logs:
            if not isinstance(log, dict):
                continue
            for turn in log.get("turns", []):
                for tc in turn.get("tool_calls", []):
                    if tc.get("tool_name") not in visual_tools:
                        continue
                    args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                    items.append({
                        "phase": log.get("phase"),
                        "turn": turn.get("turn"),
                        "tool_name": tc.get("tool_name"),
                        "reference_id": args.get("image_reference_id", "Unknown"),
                        "arguments": tc.get("arguments"),
                        "raw_result": tc.get("raw_result"),
                    })
        return items

    def _all_tool_call_records_from_logs(self, *logs: dict) -> list:
        records = []
        for log in logs:
            if not isinstance(log, dict):
                continue
            for turn in log.get("turns", []):
                for idx, tc in enumerate(turn.get("tool_calls", []), start=1):
                    records.append({
                        "phase": log.get("phase"),
                        "turn": turn.get("turn"),
                        "call_index": idx,
                        "tool_name": tc.get("tool_name"),
                        "arguments": tc.get("arguments"),
                        "raw_result": tc.get("raw_result"),
                        "error": tc.get("error"),
                    })
        return records

    def _render_phase_output_markdown(self, phase_key: str, phase_log: dict) -> str:
        header = f"<!-- {phase_key}: {phase_log.get('phase', phase_key)} -->"
        body = (phase_log.get("final_output") or "").strip()
        if not body:
            body = "_No final output captured for this phase._"
        return f"{header}\n\n{body}\n"

    def _render_phase_trace_markdown(self, phase_key: str, phase_log: dict, mapped_images: dict = None) -> str:
        mapped_images = phase_log.get("mapped_images") or mapped_images or {}
        citations = self._collect_verified_citations_from_logs(phase_log)
        visual_items = self._collect_visual_tool_outputs_from_logs(phase_log)

        lines = [
            f"# {phase_log.get('phase', phase_key)}",
            "",
            f"- Phase key: `{phase_key}`",
            f"- Agent: `{phase_log.get('agent_name', 'unknown')}`",
            f"- Session: `{phase_log.get('agent_session_id') or 'shared'}`",
            f"- Started: `{phase_log.get('start_time', 'unknown')}`",
            f"- Ended: `{phase_log.get('end_time', 'unknown')}`",
            f"- API calls: `{phase_log.get('api_call_count', 0)}`",
            f"- Prompt tokens: `{phase_log.get('total_prompt_tokens', 0)}`",
            f"- Completion tokens: `{phase_log.get('total_comp_tokens', 0)}`",
            "",
        ]

        if mapped_images:
            lines.extend(["## Mapped Images", ""])
            for ref_id, real_name in mapped_images.items():
                lines.append(f"- `{ref_id}` → `{real_name}`")
            lines.append("")

        if phase_log.get("system_prompt"):
            lines.extend(["## System Prompt", "", self._code_fence(phase_log.get("system_prompt"), "text"), ""])
        if phase_log.get("user_prompt"):
            lines.extend(["## User Prompt", "", self._code_fence(phase_log.get("user_prompt"), "text"), ""])

        lines.extend(["## Final Output Snapshot", "", self._code_fence((phase_log.get("final_output") or "").strip() or "_No final output captured._", "md"), "", "---", ""])

        lines.extend(["## Verified Citations Captured From Tool Calls", ""])
        if citations:
            for idx, citation in enumerate(citations, start=1):
                lines.append(f"{idx}. {citation}")
        else:
            lines.append("_No verified citation-tool outputs captured in this phase._")
        lines.append("")

        lines.extend(["## Image Explanations Captured From Tool Calls", ""])
        if visual_items:
            for item in visual_items:
                lines.extend([
                    f"### {item['tool_name']} — {item['reference_id']}",
                    "",
                    f"- Turn: `{item['turn']}`",
                    "",
                    "**Function query params**",
                    "",
                    self._value_code_block(item.get("arguments"), "json"),
                    "",
                    "**Tool output**",
                    "",
                    self._value_code_block(item.get("raw_result"), "text"),
                    "",
                ])
        else:
            lines.append("_No image-analysis tool outputs captured in this phase._")
            lines.append("")

        lines.extend(["## Tool Calls By Turn", ""])
        turns = phase_log.get("turns", [])
        if not turns:
            lines.append("_No turns were captured for this phase._")
        for turn in turns:
            lines.extend([f"### Turn {turn.get('turn', '?')}", ""])
            content = (turn.get("content") or "").strip()
            if content:
                lines.extend(["**Assistant content**", "", self._code_fence(content, "md"), ""])
            tool_calls = turn.get("tool_calls", [])
            if not tool_calls:
                lines.append("_No tool calls in this turn._")
                lines.append("")
                continue
            for idx, tc in enumerate(tool_calls, start=1):
                lines.extend([f"#### {idx}. `{tc.get('tool_name', 'unknown_tool')}`", ""])
                lines.extend(["**Function query params**", "", self._value_code_block(tc.get("arguments"), "json"), ""])
                lines.extend(["**Tool output**", "", self._value_code_block(tc.get("raw_result"), "text"), ""])
                if tc.get("error"):
                    lines.extend([f"**Error:** {tc.get('error')}", ""])
        return "\n".join(lines).strip() + "\n"

    def _render_tool_outputs_summary_markdown(self, case_id: str, mode: str, phase_entries: list, mapped_images: dict = None) -> str:
        logs = [phase_log for _, phase_log in phase_entries if isinstance(phase_log, dict)]
        citations = self._collect_verified_citations_from_logs(*logs)
        visual_items = self._collect_visual_tool_outputs_from_logs(*logs)
        tool_records = self._all_tool_call_records_from_logs(*logs)

        lines = [
            f"# Tool Outputs Summary for {case_id}",
            "",
            f"- Mode: `{mode}`",
            f"- Phase count: `{len(logs)}`",
            f"- Tool call count: `{len(tool_records)}`",
            "",
        ]

        if mapped_images:
            lines.extend(["## Image Reference Map", ""])
            for ref_id, real_name in mapped_images.items():
                lines.append(f"- `{ref_id}` → `{real_name}`")
            lines.append("")

        lines.extend(["## Verified Citations (Deduplicated)", ""])
        if citations:
            for idx, citation in enumerate(citations, start=1):
                lines.append(f"{idx}. {citation}")
        else:
            lines.append("_No verified citations were captured from tool calls._")
        lines.append("")

        lines.extend(["## Image Explanations", ""])
        if visual_items:
            for item in visual_items:
                lines.extend([
                    f"### {item['phase']} — {item['tool_name']} — {item['reference_id']}",
                    "",
                    "**Function query params**",
                    "",
                    self._value_code_block(item.get("arguments"), "json"),
                    "",
                    "**Tool output**",
                    "",
                    self._value_code_block(item.get("raw_result"), "text"),
                    "",
                ])
        else:
            lines.append("_No image-analysis outputs were captured._")
            lines.append("")

        lines.extend(["## Function Query Params By Tool Call", ""])
        if tool_records:
            for record in tool_records:
                lines.extend([
                    f"### {record['phase']} — Turn {record['turn']} — `{record['tool_name']}`",
                    "",
                    self._value_code_block(record.get("arguments"), "json"),
                    "",
                ])
        else:
            lines.append("_No tool calls were captured._")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _render_phase_index_markdown(self, case_id: str, mode: str, saved_files: list, mapped_images: dict = None) -> str:
        lines = [
            f"# Phase Markdown Exports for {case_id}",
            "",
            f"- Mode: `{mode}`",
            f"- Phase count: `{len(saved_files)}`",
            "",
            "## Exported Files",
            "",
        ]

        for row in saved_files:
            lines.extend([
                f"### {row['phase_name']}",
                "",
                f"- Phase key: `{row['phase_key']}`",
                f"- Output markdown: `{os.path.basename(row['output_file'])}`",
                f"- Trace markdown: `{os.path.basename(row['trace_file'])}`",
                "",
            ])

        lines.extend(["## Shared Files", "", "- `tool_outputs_summary.md`", "- `phase_index.md`", ""])

        if mapped_images:
            lines.extend(["## Image Reference Map", ""])
            for ref_id, real_name in mapped_images.items():
                lines.append(f"- `{ref_id}` → `{real_name}`")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _save_phase_markdown_exports(self, case_id: str, case_dir: str, mode: str, phase_entries: list, mapped_images: dict = None) -> dict:
        export_dir = os.path.join(case_dir, "phase_markdowns")
        os.makedirs(export_dir, exist_ok=True)

        sorted_entries = sorted(
            phase_entries,
            key=lambda item: (self._phase_sequence_number(item[0], item[1]), item[0])
        )

        saved_files = []
        for order, (phase_key, phase_log) in enumerate(sorted_entries, start=1):
            seq = self._phase_sequence_number(phase_key, phase_log)
            prefix = f"{seq:02d}" if seq != 999 else f"{order:02d}"
            slug = self._slugify(phase_log.get("phase") or phase_key)

            output_path = os.path.join(export_dir, f"{prefix}_{slug}_output.md")
            trace_path = os.path.join(export_dir, f"{prefix}_{slug}_trace.md")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self._render_phase_output_markdown(phase_key, phase_log))

            with open(trace_path, "w", encoding="utf-8") as f:
                f.write(self._render_phase_trace_markdown(phase_key, phase_log, mapped_images))

            saved_files.append({
                "phase_key": phase_key,
                "phase_name": phase_log.get("phase", phase_key),
                "output_file": output_path,
                "trace_file": trace_path,
            })

        summary_path = os.path.join(export_dir, "tool_outputs_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self._render_tool_outputs_summary_markdown(case_id, mode, sorted_entries, mapped_images))

        index_path = os.path.join(export_dir, "phase_index.md")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(self._render_phase_index_markdown(case_id, mode, saved_files, mapped_images))

        return {
            "directory": export_dir,
            "index_file": index_path,
            "tool_summary_file": summary_path,
            "files": saved_files,
        }

    def process_case_single(self, case_id: str, case_data: dict, source_dir: str, sections_str: str, case_dir: str) -> dict:
        """Executes the single-step generation mode."""
        if "References" not in sections_str and "Citations" not in sections_str:
            sections_str += "\n- References"

        prompt_template = f"""You are an expert medical researcher, clinician, and senior editor for a high-impact, peer-reviewed medical journal. Your task is to synthesize raw clinical data into a comprehensive, CARE-compliant, publication-ready clinical case report. Pure Markdown: Output the final report in strict Markdown format only. No Conversational Filler: Provide only the medical report itself.

        ### RAW CLINICAL DATA ###
        - Clinical History: {case_data.get('history')}
        - Presentation: {case_data.get('presentation')}
        - Diagnostics: {case_data.get('diagnostics')}
        - Management: {case_data.get('management')}
        - Outcome: {case_data.get('outcome')}

        ### CARE GUIDELINES & SOURCE-DRIVEN MANUSCRIPT STRUCTURE ###
        {self._care_guidance()}

        Use EXACTLY these section headings and this order. These headings come from the atom JSON metadata, with Title and References added if missing:
        {sections_str}

        ### WRITING GUIDELINES & ACADEMIC STANDARDS ###
        1. Tone & Style: Authoritative, objective, and scholarly medical tone. Use precise clinical terminology. Use a generous manuscript budget; do not compress the case merely to save tokens.
        2. Context & Background Synthesis: Embed this case within the broader medical context mainly in the Introduction and Discussion; keep patient-specific sections grounded in the source atoms.
        3. Fidelity: Preserve exact timelines, clinical values, measurements, procedures, diagnoses, outcomes, and unresolved details from the source atoms. Do not invent patient perspective, consent, or clinical facts.

        ### FIGURE RULES ###
        {self._figure_integration_guidance('{NUM_IMAGES}', '{IMAGE_IDS}')}

        ### CITATION & REFERENCE RULES ###
        {self._citation_style_guidance()}
        Tool workflow: first use `search_pubmed` to find relevant papers, then use `fetch_ama_citations` with selected DOIs, then copy the exact formatted outputs into the References list. Use at least {self.MIN_VERIFIED_REFERENCES} verified references.
        """

        prompt_template = finalize_prompt(prompt_template)
        content_payload, image_map, virtual_ids = self._prepare_multimodal_payload(prompt_template, source_dir)

        system_instruction = (
            "You are an expert medical researcher and senior editor. "
            "Write a source-section-compliant case report with at least 10 verified references, natural citation cadence, and integrated figure callouts. "
            "Use `search_pubmed` and `fetch_ama_citations` for verified references only. "
            "Do not hallucinate citations, consent, patient perspective, or clinical facts."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content_payload}
        ]

        log_data = self._execute_llm_loop(
            messages,
            TOOL_SCHEMAS,
            case_data,
            "Single_Step_Generation",
            image_map=image_map,
            agent_name="single_generation_agent",
            max_turns=14,
        )

        clinical_atoms = self._clinical_atoms_for_prompt(case_data)
        citation_items = self._collect_verified_citations_from_logs(log_data)
        citation_context_str = self._citation_bank_for_prompt(citation_items)
        audit_log, audited_markdown = self._run_quality_audit_phase(
            case_data,
            log_data["final_output"],
            sections_str,
            image_map,
            virtual_ids,
            clinical_atoms,
            citation_context_str,
        )

        repair_log = None
        final_candidate = audited_markdown
        if self._needs_citation_repair(final_candidate):
            repair_log, final_candidate = self._run_citation_repair_phase(
                case_data,
                final_candidate,
                sections_str,
                image_map,
                virtual_ids,
                clinical_atoms,
                citation_context_str,
            )

        final_markdown = self._post_process_markdown(final_candidate, image_map)

        output_file = os.path.join(case_dir, f"{case_id}_generated.md")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(final_markdown)

        phases = {
            "phase_0": log_data,
            "phase_1": audit_log,
        }
        result = {
            "mapped_images": image_map,
            "execution": log_data,
            "audit": audit_log,
            "phases": phases,
            "citation_quality": self._citation_quality_report(final_markdown),
        }
        if repair_log:
            phases["phase_2"] = repair_log
            result["citation_repair"] = repair_log
        return result

    def _run_citation_curator_phase(self, case_data: dict, sections_str: str, clinical_atoms: dict) -> tuple:
        """Uses a fresh citation-focused agent/session to build a verified citation bank."""
        citation_tool_schemas = self._schemas_for_tools([
            "search_pubmed",
            "fetch_ama_citations",
            "search_clingen_by_keyword",
            "fetch_clingen_variant_data",
        ])

        system_prompt = (
            "Phase 0: Citation Curator. Your only job is to build a verified citation bank for a medical case report. "
            "Use tools for every reference. Do not invent citations."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Build a verified citation bank for the case report that will use these exact source-driven sections:
{sections_str}

### RAW CLINICAL ATOMS ###
- History: {clinical_atoms['history']}
- Presentation: {clinical_atoms['presentation']}
- Diagnostics: {clinical_atoms['diagnostics']}
- Management: {clinical_atoms['management']}
- Outcome: {clinical_atoms['outcome']}

### TASKS ###
1. Use `search_pubmed` with strict Boolean medical queries to identify high-relevance literature.
2. Use `fetch_ama_citations` for the selected DOI list. Do not output any reference that did not come from this tool.
3. Return a verified citation bank with no fewer than {self.MIN_VERIFIED_REFERENCES} references, targeting {self.TARGET_REFERENCE_COUNT}. Include a brief placement plan for Introduction, Case Report/diagnostic discussion if needed, and Discussion.
4. Do not cite raw patient facts; plan citations only for background, diagnostic criteria, treatment rationale, or comparison with prior literature.
5. Keep citations dispersed; never plan three or more citation markers in one sentence.

Output Markdown with these headings only: `## Verified Citation Bank` and `## Citation Placement Plan`."""}
        ]

        citation_log = self._execute_llm_loop(
            messages,
            citation_tool_schemas,
            case_data,
            "Phase_0_Citation_Curation",
            image_map=None,
            agent_name="citation_curator_agent",
            max_turns=14,
        )
        citation_items = self._collect_verified_citations_from_logs(citation_log)
        citation_bank = self._citation_bank_for_prompt(citation_items)
        citation_doc = citation_log.get("final_output") or ""
        if citation_bank and citation_bank not in citation_doc:
            citation_doc = f"## Verified Citation Bank From Tool Calls\n{citation_bank}\n\n## Curator Output\n{citation_doc}"
        return citation_log, citation_doc, citation_items

    def _run_planner_phase(self, case_data: dict, source_dir: str, sections_str: str, citation_doc: str = "") -> tuple:
        clinical_atoms = self._clinical_atoms_for_prompt(case_data)
        hist_str = clinical_atoms["history"]
        pres_str = clinical_atoms["presentation"]
        diag_str = clinical_atoms["diagnostics"]
        mgmt_str = clinical_atoms["management"]
        out_str = clinical_atoms["outcome"]

        system_prompt = (
            "Phase 1: The Planner. Your core objective is to map raw clinical data atoms into a structured, "
            "section-by-section manuscript blueprint using the source metadata section order. You MUST assess visuals "
            "with tools before planning figure placement. Use the verified citation bank; use citation tools again only if it is insufficient. Zero tolerance for hallucinations."
        )

        planner_prompt = f"""### Raw Clinical Atoms
            #### History
            {hist_str}
            #### Presentation
            {pres_str}
            #### Diagnostics
            {diag_str}
            #### Management
            {mgmt_str}
            #### Outcome
            {out_str}
            ---
            ### SOURCE-DRIVEN SECTION TARGET
            The required sections below come from atom JSON metadata. Preserve these exact headings and order.
            {sections_str}

            ### CARE COMPLIANCE TARGET
            {self._care_guidance()}

            ### VERIFIED CITATION BANK / CURATOR PLAN
            {citation_doc if citation_doc else 'No separate citation curator output is available; gather verified citations with tools if needed.'}

            ### TASKS & TOOL USAGE (STRICT NO MEMORY CITATIONS)
            1. **Visual Assessment & Routing (Tool Required):** Evaluate the provided visuals and select the appropriate analysis tool based on image complexity:
               - For standard, single-modality radiological scans (e.g., MRI, CT, X-ray), use the `analyze_radiology_image` tool.
               - For complex, combined, or multi-panel images (especially figures with visible sub-labels like A, B, C, or mixed modalities), use the `analyze_composite_figure` tool and request explicit A/B panel descriptions when labels are present.
               - Do NOT use `analyze_radiology_image` for labeled composite figures when label fidelity matters, because its auto-splitting returns `panel_1`, `panel_2` in layout order rather than the figure's original labels.
               You have EXACTLY {{NUM_IMAGES}} images. Reference IDs: [{{IMAGE_IDS}}].
            2. **Literature Use:** The Citation Curator should already provide at least {self.MIN_VERIFIED_REFERENCES} verified references. Use that bank for placement planning. If it is insufficient, use `search_pubmed` and `fetch_ama_citations` to complete it.
            3. **Atom-to-Section Mapping (The Core Plan):** Strategically assign the raw clinical atoms to the required source sections. If the source has a broad "Case Report" section, fold patient information, findings, timeline, diagnostics, treatment, and outcomes into that section without inventing extra headings.
            4. **Prior-Case Evidence:** Use `assess_disease_importance` to retrieve similar prior cases when planning novelty or comparison claims. Its legacy name denotes retrieval only. The runtime excludes the source article and supplied related literature; retrieved matches do not establish rarity or a first reported case.

            ### OUTLINE REQUIREMENTS
            * **Section-by-Section Blueprint:** Outline how the article will be written. For each required source section, describe its purpose in this specific case.
            * **Fact Integration:** Explicitly map the clinical atoms into their designated source sections.
            * **Citation Placement:** Plan at least {self.MIN_VERIFIED_REFERENCES} distinct citations across the article, mainly in Introduction and Discussion. No sentence should be planned with more than two citation markers.
            * **Figure Placement:** For EVERY image, plan both (a) a main-text sentence that explicitly references the figure/panel and explains its clinical relevance and (b) the Markdown image/caption block placed immediately after that text. Captions must be compact.
            """

        content_payload, image_map, virtual_ids = self._prepare_multimodal_payload(planner_prompt, source_dir)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_payload}
        ]
        planner_log = self._execute_llm_loop(
            messages,
            TOOL_SCHEMAS,
            case_data,
            "Phase_1_Planning",
            image_map=image_map,
            agent_name="planner_agent",
            max_turns=14,
        )

        return planner_log, planner_log["final_output"], image_map, virtual_ids, clinical_atoms

    def _run_writer_phase(self, case_data: dict, planning_doc: str, sections_str: str, image_map: dict, virtual_ids: list, clinical_atoms: dict, visual_context_str: str, citation_context_str: str) -> tuple:
        ids_string = ", ".join(virtual_ids) if virtual_ids else "None"
        num_images = len(virtual_ids)

        # Clean the sections string to explicitly remove front matter for this phase.
        writer_sections_str = self._section_lines_without(sections_str, {"Title", "Keywords", "Abstract"})

        messages = [
            {"role": "system", "content": "Phase 2: The Writer. Draft the source-section-structured main body of a publication-ready academic manuscript in strict Markdown. DO NOT write front matter, but DO include inline citations and References when References is in the required section list."},
            {"role": "user", "content": f"""You are an expert medical clinician and senior editor. Write the main body of the manuscript based on the Strategic Plan & Outline and the Raw Clinical Atoms provided below.

### RAW CLINICAL ATOMS (FOR GROUNDING) ###
- History: {clinical_atoms['history']}
- Presentation: {clinical_atoms['presentation']}
- Diagnostics: {clinical_atoms['diagnostics']}
- Management: {clinical_atoms['management']}
- Outcome: {clinical_atoms['outcome']}

### STRATEGIC PLAN & OUTLINE ###
{planning_doc}

### VERIFIED CITATION BANK FROM TOOL OUTPUTS (CRITICAL) ###
Use verified citations from the supplied bank or tools only. Do not invent citations. Stage 2 must already fit no fewer than {self.MIN_VERIFIED_REFERENCES} distinct inline citations into appropriate literature-supported statements and must include a References section if listed below.
{citation_context_str}

### CLINICAL AUTHENTICITY & FIDELITY RULES (CRITICAL) ###
1. Zero Timeline Sanitization: Preserve the "messy reality" of the patient history (missed appointments, unrelated surgeries, specific intervals). Do NOT smooth over the timeline.
2. Procedural Precision: Retain the exact clinical and mechanical terminology used in the source atoms.
3. Granular Specifics: Include all specific statistical claims, precise anatomical measurements, and multi-system secondary diagnoses present in the raw atoms.

### VERBATIM VISUAL ASSETS (CRITICAL) ###
Below are the exact, expert-level interpretations of the images.
Use these image observations as supporting evidence; preserve uncertainty and check them against the visible images and clinical atoms. Preserve detailed panel breakdowns (e.g., Panel A, Panel B) in the main-text interpretation when clinically relevant, but keep the caption concise.
{visual_context_str}

### CARE, CITATION, AND FIGURE STANDARDS ###
{self._care_guidance()}

### WRITING GUIDELINES & ACADEMIC STANDARDS ###
1. Tone & Style: Authoritative, objective, and scholarly medical tone. Use a generous manuscript budget and do not compress important clinical reasoning merely to save tokens.
2. Mandatory Structure (STRICT): You MUST ONLY use the section headings provided below.
**CRITICAL: DO NOT WRITE A TITLE, KEYWORDS, OR ABSTRACT. Start directly with the first main-body section.**
{writer_sections_str}

3. Citation cadence:
{self._citation_style_guidance()}

4. Figure Integration:
{self._figure_integration_guidance(num_images, ids_string)}

Pure Markdown: Output the text in strict Markdown format only.
No Conversational Filler: Provide only the medical report itself."""}
        ]
        writer_log = self._execute_llm_loop(
            messages,
            TOOL_SCHEMAS,
            case_data,
            "Phase_2_Drafting",
            image_map=image_map,
            agent_name="writer_agent",
            max_turns=12,
        )
        return writer_log, writer_log["final_output"]

    def _run_editor_phase(self, case_data: dict, draft: str, sections_str: str, image_map: dict, virtual_ids: list, clinical_atoms: dict, visual_context_str: str, citation_context_str: str) -> tuple:
        num_images = len(virtual_ids)
        ids_string = ", ".join(virtual_ids) if virtual_ids else "None"
        front_matter_names = self._front_matter_section_names(sections_str)
        front_matter_sections_str = self._section_bullets(front_matter_names)
        post_front_matter_sections_str = self._section_lines_without(sections_str, {"Title", "Keywords", "Abstract"})
        messages = [
            {"role": "system", "content": "Phase 3: The Editor. Polish the manuscript into a source-section-compliant case report, generate required front matter including Title, preserve factual alignment, enforce at least 10 verified citations, and ensure every figure is described in the main text."},
            {"role": "user", "content": f"""Review and refine this draft for final publication.

### DRAFT TO REFINE ###
{draft}

### ORIGINAL CLINICAL ATOMS (FOR FACT-CHECKING) ###
- History: {clinical_atoms['history']}
- Presentation: {clinical_atoms['presentation']}
- Diagnostics: {clinical_atoms['diagnostics']}
- Management: {clinical_atoms['management']}
- Outcome: {clinical_atoms['outcome']}

### VERIFIED CITATION BANK FROM TOOL OUTPUTS (DO NOT INVENT REFERENCES) ###
{citation_context_str}

### VERBATIM VISUAL ASSETS (CRITICAL) ###
{visual_context_str}

### EDITORIAL STANDARDS (CRITICAL) ###
{self._care_guidance()}

1. Front Matter: You MUST create the following front-matter sections and place them at the very beginning of the manuscript. Title is mandatory at this third stage.
{front_matter_sections_str}
   - Title: Must be short, concise, highly academic, and include the phrase "case report" (e.g., "A Case Report of...") in strict adherence to CARE guidelines.
   - If Abstract is listed above, write a single continuous paragraph under 200 words. Do not include citations in the abstract.  Be concise and selective: include only key clinical facts.
2. Strict Sections: Following the front matter, include the exact main-body and back-matter sections listed below. Do not repeat front matter later:
{post_front_matter_sections_str}
3. Citation Ordering and Density: Renumber in-text citations and the final References list so they appear in first-use order (e.g., [1], then [2]), while preserving natural placement and no fewer than {self.MIN_VERIFIED_REFERENCES} unique verified references.
{self._citation_style_guidance()}
4. Fact-Check against Atoms: Cross-reference the DRAFT against the ORIGINAL CLINICAL ATOMS provided above. Correct any clinical values, timelines, or facts that drifted during drafting.
5. Verify Figures and Main-Text Callouts:
{self._figure_integration_guidance(num_images, ids_string)}
6. Perfect Formatting: Output ONLY the final perfected Markdown manuscript. Do not include conversational preambles."""}
        ]
        editor_log = self._execute_llm_loop(
            messages,
            TOOL_SCHEMAS,
            case_data,
            "Phase_3_Editing",
            image_map=image_map,
            agent_name="editor_agent",
            max_turns=12,
        )
        return editor_log, editor_log["final_output"]

    def _run_quality_audit_phase(self, case_data: dict, manuscript: str, sections_str: str, image_map: dict, virtual_ids: list, clinical_atoms: dict, citation_context_str: str = "") -> tuple:
        """Final compliance pass targeting source sections, citations, CARE substance, and figures."""
        num_images = len(virtual_ids)
        ids_string = ", ".join(virtual_ids) if virtual_ids else "None"

        messages = [
            {"role": "system", "content": "Phase 4: The Compliance Auditor. Repair only what is necessary so the manuscript preserves source-driven section order, follows CARE in substance, contains at least 10 verified citations, and references every figure in the main text before the caption."},
            {"role": "user", "content": f"""Audit and repair the manuscript below. Output only the corrected Markdown manuscript.

### MANUSCRIPT TO AUDIT ###
{manuscript}

### ORIGINAL CLINICAL ATOMS (DO NOT CONTRADICT OR EXPAND BEYOND THESE) ###
- History: {clinical_atoms['history']}
- Presentation: {clinical_atoms['presentation']}
- Diagnostics: {clinical_atoms['diagnostics']}
- Management: {clinical_atoms['management']}
- Outcome: {clinical_atoms['outcome']}

### REQUIRED FINAL SECTION ORDER ###
{sections_str}

### VERIFIED CITATION BANK FROM TOOL OUTPUTS ###
{citation_context_str}

### AUDIT TARGETS ###
1. Source-section and CARE compliance:
{self._care_guidance()}

2. Citation density and reference integrity:
{self._citation_style_guidance()}

3. Figure integration:
{self._figure_integration_guidance(num_images, ids_string)}

### REPAIR RULES ###
- Do not add new references, DOIs, PMIDs, authors, or unsupported clinical facts beyond the verified citation bank supplied above.
- Do not fabricate patient perspective, consent, ethics approval, or conflicts of interest. If absent, state that it was not available in the source material.
- If citations are packed into one sentence, redistribute them across nearby relevant statements while keeping the References list consistent.
- Ensure the final manuscript contains no fewer than {self.MIN_VERIFIED_REFERENCES} distinct inline citation numbers and no fewer than {self.MIN_VERIFIED_REFERENCES} References items. If the supplied verified bank has fewer than this, keep every verified reference and do not invent replacements.
- Ensure each figure has a main-text callout before its image block and a compact caption after it.
- Keep the output in strict Markdown with no commentary."""}
        ]

        audit_log = self._execute_llm_loop(
            messages,
            [],
            case_data,
            "Phase_4_CARE_Citation_Figure_Audit",
            image_map=image_map,
            agent_name="compliance_auditor_agent",
            max_turns=12,
        )
        return audit_log, audit_log["final_output"]

    def _run_citation_repair_phase(self, case_data: dict, manuscript: str, sections_str: str, image_map: dict, virtual_ids: list, clinical_atoms: dict, citation_context_str: str = "") -> tuple:
        """Tool-enabled repair pass when the final manuscript still has fewer than 10 citations/references."""
        citation_tool_schemas = self._schemas_for_tools(["search_pubmed", "fetch_ama_citations"])
        report = self._citation_quality_report(manuscript)
        num_images = len(virtual_ids)
        ids_string = ", ".join(virtual_ids) if virtual_ids else "None"

        messages = [
            {"role": "system", "content": "Citation Repair Agent. Use citation tools if needed, then minimally revise the manuscript so it has at least 10 verified references and matching inline citations. Do not invent references."},
            {"role": "user", "content": f"""The manuscript below failed the minimum citation audit.

### CURRENT CITATION QUALITY REPORT ###
{json.dumps(report, indent=2)}

### REQUIRED FINAL SECTION ORDER ###
{sections_str}

### EXISTING VERIFIED CITATION BANK ###
{citation_context_str}

### MANUSCRIPT TO REPAIR ###
{manuscript}

### ORIGINAL CLINICAL ATOMS ###
- History: {clinical_atoms['history']}
- Presentation: {clinical_atoms['presentation']}
- Diagnostics: {clinical_atoms['diagnostics']}
- Management: {clinical_atoms['management']}
- Outcome: {clinical_atoms['outcome']}

### REPAIR INSTRUCTIONS ###
1. If the existing verified citation bank has fewer than {self.MIN_VERIFIED_REFERENCES} usable references, first use `search_pubmed` and then `fetch_ama_citations` to add enough verified references.
2. Revise only the Introduction, literature-supported portions of Case Report/diagnostic or treatment rationale, Discussion, and References as needed.
3. Preserve all clinical facts, source-driven section order, figure image syntax, and figure callouts.
4. Final output must contain no fewer than {self.MIN_VERIFIED_REFERENCES} distinct inline citation numbers and no fewer than {self.MIN_VERIFIED_REFERENCES} numbered References entries.
5. Keep figure handling unchanged except to preserve this rule:
{self._figure_integration_guidance(num_images, ids_string)}
6. Output only the corrected Markdown manuscript."""}
        ]
        repair_log = self._execute_llm_loop(
            messages,
            citation_tool_schemas,
            case_data,
            "Phase_5_Citation_Minimum_Repair",
            image_map=image_map,
            agent_name="citation_repair_agent",
            max_turns=14,
        )
        return repair_log, repair_log["final_output"]

    def process_case_multi(self, case_id: str, case_data: dict, source_dir: str, sections_str: str, case_dir: str) -> dict:
        """Executes the multi-agent staged generation mode."""
        trajectory_log = {"mapped_images": {}, "phases": {}}

        if "References" not in sections_str and "Citations" not in sections_str:
            sections_str += "\n- References"

        clinical_atoms = self._clinical_atoms_for_prompt(case_data)

        # --- PHASE 0: CITATION CURATOR AGENT ---
        citation_log, citation_doc, citation_items = self._run_citation_curator_phase(
            case_data,
            sections_str,
            clinical_atoms,
        )
        trajectory_log["phases"]["phase_0"] = citation_log

        # --- PHASE 1: PLANNER AGENT ---
        planner_log, planning_doc, image_map, virtual_ids, clinical_atoms = self._run_planner_phase(
            case_data,
            source_dir,
            sections_str,
            citation_doc,
        )
        trajectory_log["mapped_images"] = image_map
        trajectory_log["phases"]["phase_1"] = planner_log

        all_citation_items = self._collect_verified_citations_from_logs(citation_log, planner_log)
        if not all_citation_items:
            all_citation_items = citation_items
        citation_context_str = self._citation_bank_for_prompt(all_citation_items)

        # Extract verbatim visual analyses directly from the tool execution logs.
        visual_assets = []
        for turn in planner_log.get("turns", []):
            for tc in turn.get("tool_calls", []):
                if tc.get("tool_name") in ["analyze_composite_figure", "analyze_radiology_image"]:
                    ref_id = tc.get("arguments", {}).get("image_reference_id", "Unknown")
                    raw_res = tc.get("raw_result")
                    if raw_res:
                        visual_assets.append(f"### Image {ref_id} Verbatim Analysis ###\n{self._format_tool_result_for_prompt(raw_res)}")

        visual_context_str = "\n\n".join(visual_assets) if visual_assets else "No visual tools were called."

        # --- PHASE 2: WRITER AGENT ---
        writer_log, draft = self._run_writer_phase(
            case_data,
            planning_doc,
            sections_str,
            image_map,
            virtual_ids,
            clinical_atoms,
            visual_context_str,
            citation_context_str,
        )
        trajectory_log["phases"]["phase_2"] = writer_log
        citation_context_str = self._citation_bank_for_prompt(
            self._collect_verified_citations_from_logs(citation_log, planner_log, writer_log)
        )

        # --- PHASE 3: EDITOR AGENT ---
        editor_log, refined_markdown = self._run_editor_phase(
            case_data,
            draft,
            sections_str,
            image_map,
            virtual_ids,
            clinical_atoms,
            visual_context_str,
            citation_context_str,
        )
        trajectory_log["phases"]["phase_3"] = editor_log
        citation_context_str = self._citation_bank_for_prompt(
            self._collect_verified_citations_from_logs(citation_log, planner_log, writer_log, editor_log)
        )

        # --- PHASE 4: CARE / CITATION / FIGURE AUDIT AGENT ---
        audit_log, audited_markdown = self._run_quality_audit_phase(
            case_data,
            refined_markdown,
            sections_str,
            image_map,
            virtual_ids,
            clinical_atoms,
            citation_context_str,
        )
        trajectory_log["phases"]["phase_4"] = audit_log

        final_candidate = audited_markdown
        if self._needs_citation_repair(final_candidate):
            repair_log, final_candidate = self._run_citation_repair_phase(
                case_data,
                final_candidate,
                sections_str,
                image_map,
                virtual_ids,
                clinical_atoms,
                citation_context_str,
            )
            trajectory_log["phases"]["phase_5"] = repair_log

        final_markdown = self._post_process_markdown(final_candidate, image_map)
        trajectory_log["citation_quality"] = self._citation_quality_report(final_markdown)

        output_file = os.path.join(case_dir, f"{case_id}_generated.md")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(final_markdown)

        return trajectory_log

    def process_case(self, json_path: str):
        json_path = str(Path(json_path).resolve())
        stem = Path(json_path).stem
        case_id = self._slugify(stem.removesuffix("_atoms")) or "case"
        case_dir = os.path.join(self.working_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        log_output_path = os.path.join(case_dir, f"generation_log_{self.mode}.json")

        print(f"\nProcessing Case ID: {case_id} | Mode: {self.mode.upper()} | JSON Path: {json_path}")

        # Track master timestamps
        start_dt = datetime.now()
        start_time_real = time.time()

        log_envelope = {
            "case_id": case_id,
            "mode": self.mode,
            "start_time": start_dt.isoformat(),
            "end_time": None,
            "total_time_seconds": 0,
            "total_api_calls": 0,
            "total_prompt_tokens": 0,
            "total_comp_tokens": 0,
            "status": "failed",
            "error_message": None
        }

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)

            if not isinstance(case_data, dict):
                raise ValueError("Case input must be a JSON object")
            metadata = case_data.get("metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            case_root = Path(json_path).parent
            provenance_path = case_root / "provenance.json"
            provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
            if not isinstance(provenance, dict):
                raise ValueError("provenance.json must contain a JSON object")
            supplied_source_dir = metadata.get("source_directory")
            source_path = Path(supplied_source_dir) if supplied_source_dir else case_root / "imgs"
            if not source_path.is_absolute():
                source_path = case_root / source_path
            if not supplied_source_dir and not source_path.is_dir():
                source_path = case_root
            source_dir = str(source_path.resolve())
            paper_sections = metadata.get("paper_sections_found", provenance.get("paper_sections_found", []))
            optional_sections = metadata.get("optional_sections_requested", [])
            # Only clinical atoms and runtime-only metadata survive ingestion. Source
            # titles, abstracts, authors, captions and references never enter prompts.
            exclusions = self._retrieval_exclusions({"metadata": metadata, "_provenance": provenance, **{
                key: value for key, value in case_data.items() if key in {
                    "source_identifiers", "related_exclusions", "exclusions"
                }
            }})
            case_data = {key: case_data.get(key, []) for key in (
                "history", "presentation", "diagnostics", "management", "outcome"
            )}
            case_data["metadata"] = {"source_directory": source_dir}
            case_data["_retrieval_exclusions"] = exclusions
            sections_str = self._build_care_sections_str(paper_sections, optional_sections)

            os.makedirs(case_dir, exist_ok=True)
            case_img_dir = os.path.join(case_dir, "imgs")
            os.makedirs(case_img_dir, exist_ok=True)
            if Path(source_dir).resolve() != Path(case_img_dir).resolve():
                self._copy_images(source_dir, case_img_dir)

            if self.mode == "single":
                result_data = self.process_case_single(case_id, case_data, source_dir, sections_str, case_dir)
            else:
                result_data = self.process_case_multi(case_id, case_data, source_dir, sections_str, case_dir)

            phase_entries = self._phase_entries_from_result(result_data)
            phases = [phase_log for _, phase_log in phase_entries]

            # Aggregate totals across all phases
            for p in phases:
                log_envelope["total_api_calls"] += p.get("api_call_count", 0)
                log_envelope["total_prompt_tokens"] += p.get("total_prompt_tokens", 0)
                log_envelope["total_comp_tokens"] += p.get("total_comp_tokens", 0)

            try:
                phase_markdown_exports = self._save_phase_markdown_exports(
                    case_id=case_id,
                    case_dir=case_dir,
                    mode=self.mode,
                    phase_entries=phase_entries,
                    mapped_images=result_data.get("mapped_images", {}),
                )
                result_data["phase_markdown_exports"] = phase_markdown_exports
            except Exception as export_error:
                result_data["phase_markdown_exports_error"] = str(export_error)
                print(f"[!] {case_id}: Phase markdown export warning -> {export_error}")

            log_envelope.update(result_data)
            log_envelope["status"] = "success"
            log_envelope["output_file"] = os.path.join(case_dir, f"{case_id}_generated.md")
            print(f"[+] {case_id}: {self.mode.capitalize()}-stage generation complete.")

        except Exception as e:
            log_envelope["error_message"] = str(e)
            print(f"[X] {case_id}: Error -> {e}")

        finally:
            # Mark the global end times
            log_envelope["end_time"] = datetime.now().isoformat()
            log_envelope["total_time_seconds"] = round(time.time() - start_time_real, 2)

            with open(log_output_path, "w", encoding="utf-8") as f:
                json.dump(log_envelope, f, indent=4)
        log_envelope["log_path"] = log_output_path
        return log_envelope

    def run(self):
        json_files = glob.glob(os.path.join(self.working_dir, "*", "*_atoms.json"))
        print(f"Starting journal generation for {len(json_files)} cases...")
        print(f"Model ID: {self.model_id} | Mode: {self.mode.upper()}")
        print("-" * 50)

        for path in json_files:
            self.process_case(path)

        print("-" * 50)
        print("Batch journal generation complete.")
