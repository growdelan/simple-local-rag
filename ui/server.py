"""Lokalny serwer HTTP; frontend bez frameworka i bez procesu budowania."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter

import ollama
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).with_name("static")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".epub", ".docx", ".html", ".htm", ".csv"}


class QueryRequest(BaseModel):
    collection: str = Field(min_length=3, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    rerank: bool = True
    think: bool | None = None
    model: str | None = Field(default=None, min_length=1, max_length=256)


def available_models(base_url):
    client = ollama.Client(host=base_url, timeout=5.0)
    try:
        return sorted(
            model.model
            for model in client.list().models
            if "completion" in (client.show(model.model).capabilities or [])
        )
    except (ollama.ResponseError, ConnectionError, OSError) as exc:
        raise HTTPException(
            503, "Nie można pobrać modeli. Sprawdź, czy Ollama działa."
        ) from exc
    except Exception as exc:
        raise HTTPException(
            503, "Nie udało się odczytać listy modeli z Ollamy."
        ) from exc


def create_app(backend=None):
    if backend is None:
        from ui import app as backend
    api = FastAPI(title="Local RAG", docs_url=None, redoc_url=None)
    operation_lock = Lock()
    api.state.operation_lock = operation_lock
    query_progress = {"phase": "idle"}

    def report(phase, details=None):
        nonlocal query_progress
        query_progress = {"phase": phase, **(details or {})}

    @api.get("/api/progress")
    def progress():
        return query_progress

    @contextmanager
    def exclusive():
        if not operation_lock.acquire(blocking=False):
            raise HTTPException(
                409, "Trwa już analiza lub import. Poczekaj na zakończenie."
            )
        try:
            yield
        finally:
            operation_lock.release()

    @api.middleware("http")
    async def local_requests(request: Request, call_next):
        # Domyślnie serwer jest lokalny; blokujemy zapisy ze stron zewnętrznych.
        host = request.url.hostname
        if host not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            return JSONResponse({"detail": "Niedozwolony host."}, status_code=403)
        origin = request.headers.get("origin")
        if (
            request.method not in {"GET", "HEAD"}
            and origin
            and origin != str(request.base_url).rstrip("/")
        ):
            return JSONResponse(
                {"detail": "Niedozwolone źródło żądania."}, status_code=403
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        )
        return response

    @api.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @api.get("/api/state")
    def state():
        return {
            "collections": backend.get_collection_names(),
            "model": backend.STANDARD_MODEL,
            "busy": operation_lock.locked(),
        }

    @api.get("/api/models")
    def models():
        return {
            "models": available_models(backend.OLLAMA_BASE_URL),
            "default": backend.STANDARD_MODEL,
        }

    def validate_name(name):
        try:
            return backend.validate_collection_name(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def require_collection(name):
        validate_name(name)
        if name not in backend.get_collection_names():
            raise HTTPException(404, "Nie znaleziono kolekcji. Odśwież listę.")

    @api.post("/api/query")
    def query(body: QueryRequest):
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "Wpisz pytanie.")
        with exclusive():
            require_collection(body.collection)
            model = body.model or backend.STANDARD_MODEL
            if model not in available_models(backend.OLLAMA_BASE_URL):
                raise HTTPException(
                    400, "Model jest niedostępny do rozmowy. Odśwież listę modeli."
                )
            started = perf_counter()
            report("retrieval")
            history = []
            for history, _ in backend.query_collection(
                body.collection,
                question,
                [],
                use_rerank=body.rerank,
                model_name=model,
                progress=report,
                think=body.think,
            ):
                pass
            if not history or history[-1]["role"] != "assistant":
                raise HTTPException(500, "Nie otrzymano odpowiedzi.")
            answer = history[-1]["content"]
            if answer.startswith("Error:"):
                raise HTTPException(500, answer.removeprefix("Error:").strip())
            return {
                "answer": answer,
                "seconds": round(perf_counter() - started, 2),
                "model": model,
            }

    @api.post("/api/collections")
    def upload(
        name: str = Form(...),
        enrichment: bool = Form(False),
        files: list[UploadFile] = File(...),
    ):
        try:
            with exclusive():
                validate_name(name)
                if name in backend.get_collection_names():
                    raise HTTPException(
                        409, "Ta nazwa już istnieje. Wybierz inną nazwę kolekcji."
                    )
                if not 1 <= len(files) <= 30:
                    raise HTTPException(400, "Wybierz od 1 do 30 plików.")
                with tempfile.TemporaryDirectory(prefix="local-rag-") as directory:
                    paths, names, total = [], set(), 0
                    for upload_file in files:
                        filename = (
                            (upload_file.filename or "")
                            .replace("\\", "/")
                            .split("/")[-1]
                        )
                        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                            raise HTTPException(
                                400, f"Nieobsługiwany format: {filename}"
                            )
                        if filename in names:
                            raise HTTPException(400, "Nazwy plików muszą być unikalne.")
                        names.add(filename)
                        path = Path(directory) / filename
                        with path.open("wb") as output:
                            while chunk := upload_file.file.read(1024 * 1024):
                                total += len(chunk)
                                if total > MAX_UPLOAD_BYTES:
                                    raise HTTPException(
                                        413, "Łączny limit plików to 100 MB."
                                    )
                                output.write(chunk)
                        paths.append(str(path))
                    _, message = backend.create_collection(
                        paths, name, pro_embeddings=enrichment
                    )
                    if message.startswith("Error"):
                        raise HTTPException(500, message)
                    return {"collection": name, "message": message}
        finally:
            for upload_file in files:
                upload_file.file.close()

    @api.delete("/api/collections/{name}")
    def delete(name: str):
        with exclusive():
            require_collection(name)
            _, message = backend.delete_collection(name)
            if message.startswith("Error"):
                raise HTTPException(500, message)
            return {"message": message}

    api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return api


def serve(backend):
    uvicorn.run(
        create_app(backend), host="127.0.0.1", port=int(os.getenv("PORT", "7860"))
    )
