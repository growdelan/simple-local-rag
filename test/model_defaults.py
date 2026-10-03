"""Ollama: domyślne thinking, osobny strumień i limit generacji."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import app
from llama_index.llms.ollama import Ollama


def main():
    client = Mock()
    llm = app._make_answer_llm("test", client=client)
    assert llm.model == "test" and llm.thinking is False
    with patch.object(app, "RAG_THINKING", "default"):
        assert app._make_answer_llm("test").thinking is None

    def chunks(content, reason="stop"):
        return iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": "prywatny ślad",
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                    "done_reason": reason,
                },
            ]
        )

    client.chat.return_value = chunks('{"claims": []}')
    result = "".join(
        app._stream_json(
            llm,
            app.text_qa_template,
            app.ANSWER_SCHEMA,
            query_str="Pytanie?",
            context_str="Tekst",
        )
    )
    assert result == '{"claims": []}' and "prywatny" not in result
    assert client.chat.call_args.kwargs["think"] is False
    evidence = {"S1": {"text": "Biblioteka jest otwarta o ósmej."}}
    claims = [{"evidence_id": "S1", "answer": "Otwarte o ósmej."}]
    client.chat.return_value = chunks('{"assessment": "supported", "accepted": true}')
    assert (
        app._verify_grounding(
            llm, json.dumps({"claims": claims}), evidence, "Kiedy otwarta?"
        )
        == claims
    )
    assert client.chat.call_args.kwargs["think"] is False
    assert (
        client.chat.call_args.kwargs["options"]["num_predict"] == app.VERIFY_NUM_PREDICT
    )
    for enabled in (True, False):
        with patch.object(app, "RAG_THINKING", "true" if not enabled else "false"):
            chosen = app._make_answer_llm("test", think=enabled, client=client)
        assert chosen.thinking is enabled
        client.chat.return_value = chunks('{"claims": []}')
        list(
            app._stream_json(
                chosen,
                app.text_qa_template,
                app.ANSWER_SCHEMA,
                query_str="Pytanie?",
                context_str="Tekst",
            )
        )
        assert client.chat.call_args.kwargs["think"] is enabled
        assert client.chat.call_args.kwargs["options"]["num_predict"] == (
            app.THINK_NUM_PREDICT if enabled else app.OLLAMA_NUM_PREDICT
        )
        client.chat.return_value = chunks(
            '{"assessment": "supported", "accepted": true}'
        )
        app._verify_grounding(
            chosen, json.dumps({"claims": claims}), evidence, "Kiedy?"
        )
        assert client.chat.call_args.kwargs["think"] is enabled
        assert client.chat.call_args.kwargs["options"]["num_predict"] == (
            app.THINK_VERIFY_NUM_PREDICT if enabled else app.VERIFY_NUM_PREDICT
        )
    client.chat.return_value = chunks('{"claims":', reason="length")
    try:
        list(
            app._stream_json(
                llm,
                app.text_qa_template,
                app.ANSWER_SCHEMA,
                query_str="Pytanie?",
                context_str="Tekst",
            )
        )
    except ValueError as exc:
        assert "limi" in str(exc)
    else:
        raise AssertionError("Brak obsługi limitu generacji")
    print(
        "PASS: szybki tryb bez thinking dla obu wywołań, opcja default, ukryty ślad, sygnalizacja limitu"
    )


if __name__ == "__main__":
    main()
