import json

import pytest

from medcase_agent.cli import main, read_exclusions, validate_case


def test_validate_only_requires_no_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    case = tmp_path / "demo_atoms.json"
    case.write_text(json.dumps({"history": ["Synthetic incomplete history."]}))
    assert main(["generate", str(case), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"


@pytest.mark.parametrize("data", [{}, [], {"history": [123]}, {"history": {"x": "y"}}])
def test_invalid_input_is_rejected(tmp_path, data):
    case = tmp_path / "bad.json"
    case.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_case(case)


def test_exclusions_reject_string_instead_of_identifier_list(tmp_path):
    path = tmp_path / "exclude.json"
    path.write_text(json.dumps({"pmcids": "PMC123"}))
    with pytest.raises(ValueError, match="list of strings"):
        read_exclusions(path)


def test_missing_api_config_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    case = tmp_path / "demo.json"
    case.write_text(json.dumps({"presentation": ["Synthetic observation."]}))
    with pytest.raises(SystemExit) as exc:
        main(["generate", str(case)])
    assert exc.value.code == 1
    assert "Set OPENAI_MODEL" in capsys.readouterr().err
