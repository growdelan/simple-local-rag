"""Izolowany scenariusz RAG; sztuczne dane, bez pobierania modeli."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import app
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM


def main():
    assert app._global_rerank is None, "Import nie może ładować rerankera"
    import json

    sources = {
        "S1": {
            "text": "Biblioteka jest otwarta od ósmej do szesnastej.",
            "label": "sample.txt",
        }
    }

    def render(claims, context=None):
        return app._render_grounded_answer(
            json.dumps({"claims": claims}),
            sources,
            context if context is not None else "[S1] " + sources["S1"]["text"],
        )

    valid = {
        "evidence_id": "S1",
        "answer": "Otwarte od ósmej.",
    }
    assert app._parse_plain_answer("Otwarte od ósmej. [S1]", sources) == [valid]
    for bad in ("BRAK", "Otwarte od ósmej.", "Otwarte. [S9]", "A [S1], B [S9]"):
        assert not app._parse_plain_answer(bad, sources)
    assert "sample.txt [S1]" in render([valid])
    assert render([]) == app.NO_ANSWER
    assert "Nie udało" in render([{**valid, "evidence_id": "S9"}])
    assert "Nie udało" in render([valid], "obcięty kontekst")
    assert "Nie udało" in app._render_grounded_answer('{"claims":', sources, "")
    llm_check = MockLLM(max_tokens=16)
    with patch.object(
        MockLLM,
        "stream_chat",
        return_value=iter(
            [SimpleNamespace(delta='{"assessment": "unsupported", "accepted": false}')]
        ),
    ):
        assert not app._verify_grounding(
            llm_check, json.dumps({"claims": [valid]}), sources, "Godziny?"
        )
    with patch.object(
        MockLLM,
        "stream_chat",
        return_value=iter(
            [SimpleNamespace(delta='{"assessment": "invalid", "accepted": "true"}')]
        ),
    ):
        assert not app._verify_grounding(
            llm_check, json.dumps({"claims": [valid]}), sources, "Godziny?"
        )
    # Cytujemy wyłącznie pełne zdania, które zmieściły się w kontekście.
    from llama_index.core.schema import TextNode, NodeWithScore

    source_nodes = [
        NodeWithScore(
            node=TextNode(
                text="Pierwsze zdanie. Drugie zdanie. Trzecie zdanie.",
                metadata={"file_name": "sample.txt"},
            )
        )
    ]
    budget = len(app.get_tokenizer()("[S1] Pierwsze zdanie.")) + 2
    limited, context = app._prepare_evidence(source_nodes, max_tokens=budget)
    assert limited["S1"]["quote"] == "Pierwsze zdanie."
    assert "Drugie" not in context and "Drugie" not in limited["S1"]["quote"]
    assert len(app.get_tokenizer()(context)) <= budget

    embedding = MockEmbedding(embed_dim=32)
    llm = MockLLM(max_tokens=16)
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as work:
        try:
            os.chdir(work)
            source = Path(work) / "sample.txt"
            source.write_text(
                "Biblioteka jest otwarta od ósmej do szesnastej. " * 250,
                encoding="utf-8",
            )
            with (
                patch.object(app, "OllamaEmbedding", return_value=embedding),
                patch.object(app, "Ollama", return_value=llm),
            ):
                _, status = app.create_collection([str(source)], "scenario")
                assert "successfully" in status, status
                count = app._get_chroma_client().get_collection("scenario").count()
                assert count > 4, count
                assert f"chunks: {count}" in status
                import json

                contexts = []

                def stream_answer(messages, format):
                    if format == app.VERIFY_SCHEMA:
                        yield SimpleNamespace(
                            delta='{"assessment": "supported", "accepted": true}'
                        )
                        return
                    assert format["properties"]["claims"]["items"]["properties"][
                        "evidence_id"
                    ]["enum"] == ["S1", "S2", "S3", "S4"]
                    contexts.append(messages[-1].content)
                    yield SimpleNamespace(
                        delta=json.dumps(
                            {
                                "claims": [
                                    {
                                        "evidence_id": "S1",
                                        "answer": "Biblioteka jest otwarta od ósmej do szesnastej.",
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )

                with (
                    patch.object(MockLLM, "stream_chat", side_effect=stream_answer),
                    patch.object(
                        app,
                        "_get_reranker",
                        side_effect=AssertionError("Reranker wyłączony"),
                    ),
                ):
                    result = list(
                        app.query_collection(
                            "scenario",
                            "Kiedy otwarta jest biblioteka?",
                            [],
                            use_rerank=False,
                        )
                    )
                    assert len(contexts) == 1
                    assert "[S4]" in contexts[0] and "[S5]" not in contexts[0]
                    assert "sample.txt [S1]" in result[-1][0][-1]["content"], result[-1]
                # Limit nie ujawnia draftu, nie uruchamia kolejnych prób i nie daje HTTP 500.
                from fastapi.testclient import TestClient
                from ui import server

                for limited_phase in ("generation", "verification"):
                    calls = []

                    def limited_stream(messages, format):
                        phase = (
                            "verification"
                            if format == app.VERIFY_SCHEMA
                            else "generation"
                        )
                        calls.append(phase)
                        if phase == limited_phase:
                            yield SimpleNamespace(
                                delta='{"accepted": true}',
                                raw={"done": True, "done_reason": "length"},
                            )
                        else:
                            yield from stream_answer(messages, format)

                    with (
                        patch.object(
                            MockLLM, "stream_chat", side_effect=limited_stream
                        ),
                        patch.object(
                            server,
                            "available_models",
                            return_value=[app.STANDARD_MODEL],
                        ),
                        TestClient(server.create_app(app)) as http,
                    ):
                        response = http.post(
                            "/api/query",
                            json={
                                "collection": "scenario",
                                "question": "Godziny?",
                                "rerank": False,
                                "think": True,
                            },
                        )
                        assert response.status_code == 200, response.text
                        assert "wyczerpał limit" in response.text
                        assert "sample.txt [S1]" not in response.text
                        assert calls == (
                            ["generation"]
                            if limited_phase == "generation"
                            else ["generation", "verification"]
                        )
                        assert http.get("/api/progress").json()["phase"] == "done"

                from unittest.mock import Mock

                reranker = Mock()
                reranker.postprocess_nodes.side_effect = lambda nodes, **kw: nodes[:4]
                with (
                    patch.object(app, "_get_reranker", return_value=reranker),
                    patch.object(MockLLM, "stream_chat", side_effect=stream_answer),
                ):
                    result = list(
                        app.query_collection(
                            "scenario", "Godziny otwarcia?", [], use_rerank=True
                        )
                    )
                    assert reranker.postprocess_nodes.call_count == 1
                    assert len(
                        reranker.postprocess_nodes.call_args_list[0].args[0]
                    ) == min(count, app.RERANK_CANDIDATES)
                    assert "sample.txt [S1]" in result[-1][0][-1]["content"], result[-1]
                # Jedna próba ratunkowa; nawet VERIFY_ANSWERS=false nie omija jej kontroli.
                for accepted in (True, False):
                    calls = []

                    def retry_stream(messages, format):
                        calls.append(format)
                        if format is None:
                            output = "Otwarte od ósmej. [S1]"
                        elif format == app.VERIFY_SCHEMA:
                            output = json.dumps(
                                {"assessment": "test", "accepted": accepted}
                            )
                        else:
                            output = '{"claims": []}'
                        yield SimpleNamespace(delta=output)

                    with (
                        patch.object(MockLLM, "stream_chat", side_effect=retry_stream),
                        patch.object(app, "VERIFY_ANSWERS", False),
                    ):
                        result = list(
                            app.query_collection(
                                "scenario", "Godziny?", [], use_rerank=False
                            )
                        )
                    assert len(calls) == 3 and calls[1] is None
                    answer = result[-1][0][-1]["content"]
                    assert ("sample.txt [S1]" in answer) == accepted
                    if not accepted:
                        assert answer == app.NO_ANSWER
                # Ponowny import zastępuje kolekcję, nie podwaja nodów.
                _, status = app.create_collection([str(source)], "scenario")
                assert "successfully" in status, status
                assert (
                    app._get_chroma_client().get_collection("scenario").count() == count
                )
                print(
                    f"PASS: {count} nodów; Chroma, oba tryby, streaming, reset kolekcji"
                )
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    main()
