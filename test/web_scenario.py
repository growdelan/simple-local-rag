"""Test HTTP bez modeli: CRUD, blokada, walidacja i lokalny frontend."""

import sys
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from ui import app, server


def main():
    with patch.object(server.ollama, "Client") as factory:
        client = factory.return_value
        client.list.return_value.models = [
            SimpleNamespace(model="chat"),
            SimpleNamespace(model="embed"),
        ]
        client.show.side_effect = [
            SimpleNamespace(capabilities=["completion"]),
            SimpleNamespace(capabilities=["embedding"]),
        ]
        assert server.available_models("http://localhost:11434") == ["chat"]
        client.list.side_effect = ConnectionError("offline")
        try:
            server.available_models("http://localhost:11434")
        except server.HTTPException as exc:
            assert exc.status_code == 503
        else:
            raise AssertionError("Brak błędu połączenia")
    names = ["sample"]

    def create(files, name, pro_embeddings=False):
        assert Path(files[0]).name == "notes.txt"
        assert Path(files[0]).read_text() == "Test dokumentu."
        names.append(name)
        return {}, "created"

    def delete(name):
        names.remove(name)
        return {}, "deleted"

    used_models = []

    def query(name, question, history, use_rerank=True, model_name=None, progress=None):
        progress("generation", {"characters": 10})
        used_models.append(model_name)
        assert name == "sample"
        assert question == "Pytanie?"
        yield [{"role": "assistant", "content": "Odpowiedź ze źródła."}], ""

    backend = SimpleNamespace(
        STANDARD_MODEL="test-model",
        OLLAMA_BASE_URL="http://localhost:11434",
        get_collection_names=lambda: names,
        validate_collection_name=app.validate_collection_name,
        create_collection=create,
        delete_collection=delete,
        query_collection=query,
    )
    api = server.create_app(backend)
    with (
        patch.object(
            server, "available_models", return_value=["test-model", "other-model"]
        ),
        TestClient(api) as client,
    ):
        page = client.get("/")
        assert page.status_code == 200 and "question-form" in page.text
        assert (
            "gradio" not in page.text.lower() and "reasoning_checkbox" not in page.text
        )
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/state").json()["collections"] == ["sample"]
        assert client.get("/api/models").json()["models"] == [
            "test-model",
            "other-model",
        ]
        body = {"collection": "sample", "question": " Pytanie? ", "rerank": False}
        assert (
            client.post("/api/query", json=body).json()["answer"]
            == "Odpowiedź ze źródła."
        )
        assert used_models[-1] == "test-model"
        assert client.get("/api/progress").json() == {
            "phase": "generation",
            "characters": 10,
        }
        response = client.post("/api/query", json={**body, "model": "other-model"})
        assert response.status_code == 200 and response.json()["model"] == "other-model"
        assert used_models[-1] == "other-model"
        assert (
            client.post("/api/query", json={**body, "model": "missing"}).status_code
            == 400
        )
        with patch.object(
            server, "available_models", side_effect=server.HTTPException(503, "Offline")
        ):
            assert client.get("/api/models").status_code == 503
            assert client.post("/api/query", json=body).status_code == 503
        assert (
            client.post("/api/query", json={**body, "question": "  "}).status_code
            == 400
        )
        assert (
            client.post("/api/query", json={**body, "collection": "absent"}).status_code
            == 404
        )
        assert (
            client.post(
                "/api/query", json=body, headers={"Origin": "https://external.example"}
            ).status_code
            == 403
        )
        assert (
            client.get("/api/state", headers={"Host": "external.example"}).status_code
            == 403
        )
        api.state.operation_lock.acquire()
        try:
            assert client.post("/api/query", json=body).status_code == 409
        finally:
            api.state.operation_lock.release()

        def upload(name, filename="notes.txt"):
            return client.post(
                "/api/collections",
                data={"name": name},
                files={"files": (filename, b"Test dokumentu.", "text/plain")},
            )

        assert upload("../outside").status_code == 400
        assert upload("sample").status_code == 409
        assert upload("new", "script.exe").status_code == 400
        with patch.object(server, "MAX_UPLOAD_BYTES", 4):
            assert upload("new").status_code == 413
        assert upload("new").status_code == 200
        assert "new" in names
        assert client.delete("/api/collections/new").status_code == 200
        assert client.delete("/api/collections/new").status_code == 404
        assert names == ["sample"]
    from llama_index.core.embeddings import MockEmbedding

    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            with patch.object(
                app, "OllamaEmbedding", return_value=MockEmbedding(embed_dim=32)
            ):
                with TestClient(server.create_app(app)) as client:
                    response = client.post(
                        "/api/collections",
                        data={"name": "http-scenario"},
                        files={
                            "files": (
                                "document.txt",
                                "Biblioteka jest otwarta o ósmej.".encode(),
                                "text/plain",
                            )
                        },
                    )
                    assert response.status_code == 200, response.text
                    assert (
                        app._get_chroma_client().get_collection("http-scenario").count()
                        == 1
                    )
                    assert (
                        client.delete("/api/collections/http-scenario").status_code
                        == 200
                    )
                    assert not Path("data/http-scenario").exists()
        finally:
            os.chdir(previous)
    print("PASS: frontend, query, upload, delete, validation, concurrency, same-origin")


if __name__ == "__main__":
    main()
