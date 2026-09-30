from types import SimpleNamespace as NS
from unittest.mock import Mock

from medcase_agent.utils import generate_llm_response, extract_json_from_text


def test_streamed_tool_arguments_and_usage_are_preserved(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "1000")
    monkeypatch.setenv("OPENAI_MAX_TOKENS_PARAM", "max_completion_tokens")
    monkeypatch.setenv("OPENAI_SEND_TEMPERATURE", "false")
    usage = NS(prompt_tokens=10, completion_tokens=20)
    chunks = [
        NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[
            NS(index=0, id="call_1", function=NS(name="search_pubmed", arguments='{"query":'))
        ]))]),
        NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[
            NS(index=0, id=None, function=NS(name=None, arguments='"example"}'))
        ]))]),
        NS(usage=usage, choices=[]),
    ]
    create = Mock(return_value=iter(chunks))
    client = NS(chat=NS(completions=NS(create=create)))
    result = generate_llm_response(client, "test-model", [], stream=True, temperature=0.2)
    assert result["tool_calls"][0]["function"]["arguments"] == '{"query":"example"}'
    assert result["usage"] is usage
    kwargs = create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 1000
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


def test_json_parser_accepts_fences_and_braces_in_strings():
    assert extract_json_from_text('```json\n{"history":["value {x}"]}\n```') == {"history": ["value {x}"]}
    assert extract_json_from_text("not a JSON answer") is None
