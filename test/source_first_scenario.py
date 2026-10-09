"""Source-first HTTP flow on a temporary Chroma index; no live LLM required."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from llama_index.core.embeddings import MockEmbedding
from ui import app, server


def scenario():
    document = Path("hours.txt")
    document.write_text("Biblioteka jest otwarta od ósmej. Zamknięcie o siedemnastej.")
    with patch.object(app, "OllamaEmbedding", return_value=MockEmbedding(embed_dim=32)):
        _, message = app.create_collection(
            [str(document)], "sample", pro_embeddings=False
        )
        assert not message.startswith("Error"), message
        collection = app._get_chroma_client().get_collection("sample")
        assert collection.count() == 1
        api = server.create_app(app)
        calls = []

        def stream(llm, template, schema, **kwargs):
            calls.append((llm.model, llm.thinking, kwargs))
            assert kwargs["query_str"] == "Kiedy otwiera się biblioteka?"
            source = kwargs.get("context_str", kwargs.get("quote", ""))
            assert "od ósmej" in source
            if schema is app.VERIFY_SCHEMA:
                yield '{"assessment":"Godzina jest podana.","accepted":true}'
            else:
                yield json.dumps(
                    {"claims": [{"evidence_id": "S1", "answer": "Od ósmej."}]}
                )

        body = {
            "collection": "sample",
            "question": "Kiedy otwiera się biblioteka?",
            "rerank": False,
        }
        with TestClient(api) as client:
            with (
                patch.object(
                    app,
                    "_make_answer_llm",
                    side_effect=AssertionError("Search must not construct LLM"),
                ),
                patch.object(
                    server,
                    "available_models",
                    side_effect=AssertionError("Search must not require chat models"),
                ),
            ):
                found = client.post("/api/search", json=body)
                assert found.status_code == 200, found.text
                found = found.json()
                assert len(found["sources"]) == 1
                assert found["sources"][0]["text"] == document.read_text()
                assert found["sources"][0]["label"] == "hours.txt"
                other = client.post(
                    "/api/search", json={**body, "question": "Inne pytanie"}
                ).json()
                assert found["search_id"] != other["search_id"]
                assert (
                    client.post(
                        "/api/search", json={**body, "question": " "}
                    ).status_code
                    == 400
                )
                assert (
                    client.post(
                        "/api/search", json={**body, "collection": "missing"}
                    ).status_code
                    == 404
                )
            answer = {
                "search_id": found["search_id"],
                "model": "test-model",
                "think": True,
            }
            with (
                patch.object(
                    server,
                    "available_models",
                    return_value=["test-model", "second-model"],
                ),
                patch.object(
                    app,
                    "search_collection",
                    side_effect=AssertionError("Answer must not retrieve"),
                ),
                patch.object(app, "_stream_json", side_effect=stream),
            ):
                for model, think in [("test-model", True), ("second-model", False)]:
                    result = client.post(
                        "/api/answer", json={**answer, "model": model, "think": think}
                    )
                    assert result.status_code == 200, result.text
                    assert result.json()["answer"].startswith("Od ósmej.")
                    assert result.json()["context_trimmed"] is False
                    assert all(c[0] == model and c[1] is think for c in calls[-2:])
                assert (
                    len(calls) == 4
                )  # Draft + verification per click, never during search.
                assert (
                    client.post(
                        "/api/answer", json={**answer, "model": "absent"}
                    ).status_code
                    == 400
                )
                with patch.object(
                    app,
                    "_stream_json",
                    side_effect=app.GenerationLimitError("Limit testowy"),
                ):
                    assert client.post("/api/answer", json=answer).status_code == 422
                assert (
                    client.post("/api/answer", json=answer).status_code == 200
                )  # Snapshot survives failure.
                assert (
                    client.post(
                        "/api/answer", json={**answer, "search_id": "missing"}
                    ).status_code
                    == 410
                )
                api.state.operation_lock.acquire()
                try:
                    assert client.post("/api/answer", json=answer).status_code == 409
                    assert client.post("/api/search", json=body).status_code == 409
                finally:
                    api.state.operation_lock.release()
                for endpoint, payload in [("search", body), ("answer", answer)]:
                    assert (
                        client.post(
                            f"/api/{endpoint}",
                            json=payload,
                            headers={"Origin": "https://elsewhere.example"},
                        ).status_code
                        == 403
                    )
            with patch.object(server, "SEARCH_TTL_SECONDS", 0):
                assert client.post("/api/answer", json=answer).status_code == 410
            with patch.object(server, "MAX_SEARCH_RESULTS", 1):
                first = client.post("/api/search", json=body).json()
                second = client.post("/api/search", json=body).json()
                assert (
                    client.post(
                        "/api/answer", json={"search_id": first["search_id"]}
                    ).status_code
                    == 410
                )
            with patch.object(app, "search_collection", return_value=[]):
                empty = client.post("/api/search", json=body).json()
                assert empty["sources"] == [] and empty["search_id"] is None
            assert collection.count() == 1
            assert client.delete("/api/collections/sample").status_code == 200
            assert (
                client.post(
                    "/api/answer", json={"search_id": second["search_id"]}
                ).status_code
                == 410
            )
            assert not Path("data/sample").exists()
    print(
        "PASS: search without chat model, frozen evidence/question, on-demand model/Think, verification, retry after failure, TTL/limit/delete invalidation, concurrency, same-origin, Chroma count=1"
    )


if __name__ == "__main__":
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            scenario()
        finally:
            os.chdir(previous)
