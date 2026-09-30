import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medcase_agent.generation import GenerationPipeline
from medcase_agent.tools.registry import TOOL_SCHEMAS
from medcase_agent.tools.pubmed_tools import _excluded, fetch_ama_citations


def completion(content="Synthetic draft", calls=None):
    return {"content": content, "tool_calls": calls or [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}}


def tool_call(name, **arguments):
    return {"id": "call_test", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class GenerationTests(unittest.TestCase):
    def test_identifier_urls_versions_and_doi_encoding_share_rag_normalization(self):
        pipeline = GenerationPipeline("unused", model_id="test-model", client=object())
        exclusions = pipeline._retrieval_exclusions({"_provenance": {"source_identifiers": {
            "pmcid": "https://pmc.ncbi.nlm.nih.gov/articles/PMC000123.1/",
            "pmid": "https://pubmed.ncbi.nlm.nih.gov/000456/",
            "doi": "https://doi.org/10.1234%2FSOURCE",
        }}})
        self.assertEqual(exclusions, {
            "exclude_pmcids": ["PMC123"], "exclude_pmids": ["456"], "exclude_dois": ["10.1234/source"]
        })
        self.assertTrue(_excluded({"pmcid": "PMC123.2"}, exclusions))
        self.assertTrue(_excluded({"pmid": "PMID: 456"}, exclusions))
        self.assertTrue(_excluded({"doi": "DOI: 10.1234/SOURCE"}, exclusions))

    def test_empty_or_unavailable_citations_never_enter_verified_bank(self):
        pipeline = GenerationPipeline("unused", model_id="test-model", client=object())
        with patch("medcase_agent.tools.pubmed_tools.requests.get") as get:
            citations = fetch_ama_citations(["10.1234/source"], exclude_dois=["10.1234/source"])
        get.assert_not_called()
        self.assertEqual(citations, "")
        self.assertEqual(pipeline._flatten_citation_items(citations), [])
        self.assertEqual(pipeline._flatten_citation_items("No verified citations available."), [])

    def test_full_mock_run_excludes_source_and_keeps_provenance_out_of_prompts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input"
            source.mkdir()
            (source / "imgs").mkdir()
            atoms = {name: ["Synthetic atom"] for name in ("history", "presentation", "diagnostics", "management", "outcome")}
            atoms.update({"title": "FORBIDDEN_SOURCE_TITLE", "abstract": "FORBIDDEN_SOURCE_ABSTRACT"})
            (source / "case_atoms.json").write_text(json.dumps(atoms))
            (source / "provenance.json").write_text(json.dumps({
                "source_identifiers": {"pmcid": "PMC11", "pmid": "22", "doi": "10.1234/source"},
                "related_exclusions": {"pmids": ["33"]},
                "title": "FORBIDDEN_PROVENANCE_TITLE",
            }))
            pipeline = GenerationPipeline(str(root / "runs"), model_id="test-model", client=object(), tools_config={"exclusions": {"dois": ["10.1234/related"]}})
            responses = [
                completion("Citation bank"),
                completion(calls=[tool_call("assess_disease_importance", diseases=["synthetic"], exclude_pmids=[])]),
                completion("Synthetic plan"), completion(), completion(), completion(), completion(),
            ]
            with patch("medcase_agent.generation.generate_llm_response", side_effect=responses) as llm, patch.dict(
                "medcase_agent.generation.AVAILABLE_TOOLS", {"assess_disease_importance": lambda **args: captured.append(args) or {"retrieved_similar_cases": []}}
            ):
                captured = []
                result = pipeline.process_case(str(source / "case_atoms.json"))
            self.assertEqual(result["status"], "success", result.get("error_message"))
            self.assertTrue(Path(result["output_file"]).is_file())
            self.assertTrue(Path(result["log_path"]).is_file())
            self.assertEqual(pipeline.mode, "multi")
            self.assertEqual(captured[0]["exclude_pmcids"], ["PMC11"])
            self.assertEqual(captured[0]["exclude_pmids"], ["22", "33"])
            self.assertEqual(captured[0]["exclude_dois"], ["10.1234/related", "10.1234/source"])
            prompts = json.dumps([call.kwargs["messages"] for call in llm.call_args_list])
            self.assertNotIn("FORBIDDEN", prompts)
            self.assertIn("Phase 3: The Editor", prompts)
            self.assertNotIn("source_identifiers", prompts)
            phases = list(result["phases"].values())
            sessions = [phase["agent_session_id"] for phase in phases]
            self.assertEqual(len(sessions), len(set(sessions)))
            starts = [call.kwargs["messages"][0]["content"] for call in llm.call_args_list]
            self.assertEqual(sum(value.startswith("Phase 2:") for value in starts), 1)
            for call in llm.call_args_list:
                system = call.kwargs["messages"][0]["content"]
                if system.startswith(("Phase 2:", "Phase 3:")):
                    self.assertEqual(call.kwargs["tools"], TOOL_SCHEMAS)
                    self.assertEqual([message["role"] for message in call.kwargs["messages"]], ["system", "user"])

    def test_disallowed_tool_is_not_executed(self):
        pipeline = GenerationPipeline("unused", model_id="test-model", client=object())
        with patch("medcase_agent.generation.generate_llm_response", side_effect=[
            completion(calls=[tool_call("search_pubmed", query="test")]), completion("done")
        ]), patch.dict("medcase_agent.generation.AVAILABLE_TOOLS", {"search_pubmed": lambda **args: self.fail("Disallowed tool executed")}):
            result = pipeline._execute_llm_loop([], [], {}, "audit")
        self.assertIn("not registered", result["turns"][0]["tool_calls"][0]["error"])

    def test_exhausted_tools_do_not_become_manuscript(self):
        pipeline = GenerationPipeline("unused", model_id="test-model", client=object())
        with patch("medcase_agent.generation.generate_llm_response", return_value=completion(calls=[tool_call("search_pubmed", query="test")])):
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                pipeline._execute_llm_loop([], [], {}, "audit", max_turns=1)

    def test_missing_input_writes_failure_log(self):
        with tempfile.TemporaryDirectory() as temp:
            result = GenerationPipeline(temp, model_id="test-model", client=object()).process_case(str(Path(temp) / "missing.json"))
            self.assertEqual(result["status"], "failed")
            self.assertTrue(Path(result["log_path"]).exists())


if __name__ == "__main__":
    unittest.main()
