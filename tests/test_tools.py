import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from medcase_agent.tools import clingen_tools, medgemma_tools, pubmed_tools


class ToolTests(unittest.TestCase):
    def test_pubmed_removes_pmid_pmcid_and_doi_before_returning_prose(self):
        response = Mock()
        response.json.return_value = {"esearchresult": {"idlist": ["1", "2", "3", "4"]}}
        records = [
            {"pmid": "2", "pmcid": "PMC20", "doi": None, "title": "hidden", "abstract": "hidden"},
            {"pmid": "3", "pmcid": "PMC30", "doi": "10.1234/source", "title": "hidden", "abstract": "hidden"},
            {"pmid": "4", "pmcid": "PMC40", "doi": "10.1234/allowed", "title": "allowed", "abstract": "allowed", "journal": "Test", "year": 2020},
        ]
        with patch.object(pubmed_tools.requests, "get", return_value=response), patch.object(pubmed_tools, "fetch_pubmed_details", return_value=records) as fetch:
            result = pubmed_tools.search_pubmed("test", exclude_pmids=["1"], exclude_pmcids=["PMC20"], exclude_dois=["https://doi.org/10.1234/source"])
        self.assertEqual(fetch.call_args.args[0], ["2", "3", "4"])
        self.assertIn("allowed", result)
        self.assertNotIn("hidden", result)

    def test_source_doi_is_never_resolved(self):
        with patch.object(pubmed_tools.requests, "get") as get:
            result = pubmed_tools.fetch_ama_citations(["10.1234/source"], exclude_dois=["10.1234/source"])
        get.assert_not_called()
        self.assertEqual(result, "")

    def test_clingen_accepts_runtime_context(self):
        response = Mock()
        response.json.return_value = {"esearchresult": {"idlist": []}}
        with patch.object(clingen_tools.requests, "get", return_value=response):
            result = clingen_tools.search_clingen_by_keyword("test", case_data={}, execution_log={}, exclude_pmids=[])
        self.assertEqual(result["status"], "error")

    def test_basic_import_does_not_load_visual_model_packages(self):
        code = "import sys; import medcase_agent.tools.registry; assert not {'torch','transformers','cv2','PIL'} & set(sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_server_image_route_needs_no_local_model(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "image.png").write_bytes(b"synthetic image bytes")
            response = Mock()
            response.json.return_value = {"choices": [{"message": {"content": "Synthetic observation"}}]}
            with patch.object(medgemma_tools.requests, "post", return_value=response) as post, patch.object(medgemma_tools, "get_medgemma_pipe", side_effect=AssertionError("local model started")):
                result = medgemma_tools.analyze_radiology_image("IMG_TEST", case_data={"metadata": {"source_directory": temp}}, execution_log={"mapped_images": {"IMG_TEST": "image.png"}}, vllm_url="https://vision.example/v1", vllm_model="test-model")
            self.assertEqual(json.loads(result)["analysis"], "Synthetic observation")
            self.assertEqual(post.call_args.args[0], "https://vision.example/v1/chat/completions")
            self.assertEqual(post.call_args.kwargs["json"]["model"], "test-model")

    def test_unconfigured_medgemma_does_not_load_or_download(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "image.png").write_bytes(b"synthetic image bytes")
            with patch.dict(os.environ, {}, clear=True), patch.object(medgemma_tools, "get_medgemma_pipe", side_effect=AssertionError("model started")), patch.object(medgemma_tools.requests, "post", side_effect=AssertionError("network used")):
                result = medgemma_tools.analyze_radiology_image("IMG_TEST", case_data={"metadata": {"source_directory": temp}}, execution_log={"mapped_images": {"IMG_TEST": "image.png"}})
            self.assertIn("not configured", json.loads(result)["error"])


if __name__ == "__main__":
    unittest.main()
