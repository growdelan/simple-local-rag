"""Controlled source/model experiment. Does not modify app code or collections.

Oracle excerpts are loaded from local EPUB files using frozen offsets and hashes.
Results contain model answers and timings, never the source passages or thinking.
Exit code 0 means completion, not semantic correctness: review every claim.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import random
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ollama
from llama_index.core import SimpleDirectoryReader
from ui import app


class MeteredClient:
    def __init__(self, client):
        self.client = client
        self.calls = []
        self.stage = "draft"

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, **kwargs):
        started = perf_counter()
        row = {
            "stage": self.stage,
            "num_predict": kwargs["options"]["num_predict"],
            "done_reason": None,
            "content": "",
            "thinking_characters": 0,
        }
        stream = None
        try:
            stream = self.client.chat(**kwargs)
            for chunk in stream:
                raw = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                row["content"] += raw["message"].get("content") or ""
                row["thinking_characters"] += len(raw["message"].get("thinking") or "")
                if raw.get("done"):
                    for key in ("done_reason", "prompt_eval_count", "eval_count"):
                        row[key] = raw.get(key)
                    for key in (
                        "load_duration",
                        "prompt_eval_duration",
                        "eval_duration",
                    ):
                        row[key + "_seconds"] = (raw.get(key) or 0) / 1e9
                yield chunk
        finally:
            if stream is not None and hasattr(stream, "close"):
                stream.close()
            row["seconds"] = perf_counter() - started
            self.calls.append(row)


def text_only(rendered):
    return "\n".join(
        part.split("\n\nUzasadnienie: ")[0] for part in rendered.split("\n\n---\n\n")
    )


def run_answer(model, question, evidence, context, client):
    """Mirror the existing draft -> verification -> conditional retry policy."""
    meter = MeteredClient(client)
    llm = app._make_answer_llm(model, think=False, client=meter)
    schema = deepcopy(app.ANSWER_SCHEMA)
    schema["properties"]["claims"]["items"]["properties"]["evidence_id"]["enum"] = list(
        evidence
    )
    started = perf_counter()
    result = {
        "draft_answer": None,
        "after_verification": None,
        "retry_answer": None,
        "answer": None,
        "error": None,
    }
    try:
        raw = "".join(
            app._stream_json(
                llm,
                app.text_qa_template,
                schema,
                query_str=question,
                context_str=context,
            )
        )
        draft = app._render_grounded_answer(raw, evidence, context)
        result["draft_answer"] = text_only(draft)
        if app.VERIFY_ANSWERS and draft != app.UNVERIFIED_ANSWER:
            meter.stage = "verify_draft"
            claims = app._verify_grounding(llm, raw, evidence, question)
            draft = app._render_grounded_answer(
                json.dumps({"claims": claims}), evidence, context
            )
        result["after_verification"] = text_only(draft)
        if draft in (app.NO_ANSWER, app.UNVERIFIED_ANSWER):
            meter.stage = "retry"
            retry_llm = llm.model_copy(
                update={"system_prompt": app.RETRY_SYSTEM_PROMPT}
            )
            retry = "".join(
                app._stream_json(
                    retry_llm,
                    app.RETRY_TEMPLATE,
                    None,
                    query_str=question,
                    context_str=context,
                )
            )
            result["retry_answer"] = retry
            claims = app._parse_plain_answer(retry, evidence)
            if claims:
                meter.stage = "verify_retry"
                claims = app._verify_grounding(
                    llm, json.dumps({"claims": claims}), evidence, question
                )
                draft = app._render_grounded_answer(
                    json.dumps({"claims": claims}), evidence, context
                )
        result["answer"] = text_only(draft)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = perf_counter() - started
    result["calls"] = meter.calls
    return result


def load_oracles(manifest, data_directory):
    books = {}
    contexts = {}
    for case in manifest["cases"]:
        evidence = {}
        for number, source in enumerate(case["sources"], 1):
            filename = source["file"]
            if Path(filename).name != filename:
                raise ValueError("Source must be a plain filename")
            if filename not in books:
                docs = SimpleDirectoryReader(
                    input_files=[str(data_directory / filename)]
                ).load_data()
                if len(docs) != 1:
                    raise ValueError("Expected one EPUB document per file")
                books[filename] = docs[0].text
            text = books[filename][source["start"] : source["end"]]
            if hashlib.sha256(text.encode()).hexdigest() != source["sha256"]:
                raise ValueError(
                    f"Source changed for {case['id']}; refusing silent substitution"
                )
            evidence[f"S{number}"] = {"text": text, "quote": text, "label": filename}
        context = "\n\n".join(
            f"[{key}] {value['text']}" for key, value in evidence.items()
        )
        tokens = len(app.get_tokenizer()(context))
        budget = (
            app.OLLAMA_NUM_CTX
            - app.OLLAMA_NUM_PREDICT
            - len(app.get_tokenizer()(app.RAG_SYSTEM_PROMPT + case["question"]))
            - 400
        )
        if tokens > budget:
            raise ValueError(f"Oracle would exceed app context budget for {case['id']}")
        contexts[case["id"]] = (evidence, context)
    return contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("oracle_cases.json")
    )
    parser.add_argument("--data", type=Path, default=Path("data/1Q84_full"))
    parser.add_argument(
        "--models", nargs="+", default=["gemma4:e2b-it-qat", "ornith-1.5:9b"]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--contexts",
        type=Path,
        help="Private frozen retrieval contexts instead of oracle sources",
    )
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("Use a new output path and positive repeats")
    manifest = json.loads(args.manifest.read_text())
    if args.case:
        if set(args.case) - {case["id"] for case in manifest["cases"]}:
            parser.error("Unknown case")
        manifest["cases"] = [
            case for case in manifest["cases"] if case["id"] in args.case
        ]
    if args.contexts:
        frozen = json.loads(args.contexts.read_text())
        contexts = {
            key: (value["evidence"], value["context"]) for key, value in frozen.items()
        }
    else:
        contexts = load_oracles(manifest, args.data)
    client = ollama.Client(host=app.OLLAMA_BASE_URL, timeout=app.RAG_TIMEOUT)
    installed = {model.model: model.digest for model in client.list().models}
    for name in args.models:
        if name not in installed:
            parser.error(f"Model not installed: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    for repeat in range(1, args.repeats + 1):
        model_order = args.models if repeat % 2 else list(reversed(args.models))
        for model in model_order:
            # M1 shares RAM with GPU; never retain the other benchmark model.
            for other in args.models:
                if other != model:
                    client.generate(model=other, prompt="", keep_alive=0)
            warm_started = perf_counter()
            warm = client.chat(
                model=model,
                messages=[{"role": "user", "content": "Napisz OK."}],
                think=False,
                options={"num_ctx": app.OLLAMA_NUM_CTX, "num_predict": 8},
                keep_alive="10m",
            )
            warm_info = {
                "seconds": perf_counter() - warm_started,
                "load_seconds": (warm.load_duration or 0) / 1e9,
            }
            print(f"MODEL {model} round={repeat} warmup={warm_info}", flush=True)
            order = list(manifest["cases"])
            random.Random(20261004 + repeat).shuffle(order)
            for case in order:
                evidence, context = contexts[case["id"]]
                result = run_answer(model, case["question"], evidence, context, client)
                row = {
                    "id": case["id"],
                    "question": case["question"],
                    "model": model,
                    "model_digest": installed[model],
                    "repeat": repeat,
                    "think": False,
                    "source_mode": "retrieved" if args.contexts else "oracle",
                    "manifest_sha256": manifest_hash,
                    "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                    "context_tokens_approx": len(app.get_tokenizer()(context)),
                    "warmup": warm_info,
                    **result,
                }
                with args.output.open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "id",
                                "model",
                                "repeat",
                                "answer",
                                "error",
                                "seconds",
                            )
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
