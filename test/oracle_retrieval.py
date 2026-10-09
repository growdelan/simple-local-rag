"""Freeze production retrieval contexts privately, without generation or DB writes."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("oracle_cases.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Private JSON path, e.g. .codex/oracle-eval/retrieved.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    manifest = json.loads(args.manifest.read_text())
    collection = app._get_chroma_client().get_collection(manifest["collection"])
    before = collection.count()
    original_prepare = app._prepare_evidence
    contexts = {}
    for case in manifest["cases"]:
        captured = {}

        def capture(nodes, max_tokens=None):
            evidence, context = original_prepare(nodes, max_tokens=max_tokens)
            captured.update(
                evidence=evidence,
                context=context,
                node_ids=[item.node.node_id for item in nodes],
            )
            return evidence, context

        def empty_answer(llm, template, schema, **kwargs):
            yield '{"claims":[]}' if schema is not None else "BRAK"

        started = perf_counter()
        with (
            patch.object(app, "_prepare_evidence", side_effect=capture),
            patch.object(app, "_stream_json", side_effect=empty_answer),
            patch.object(app, "DEBUG_CONTEXT", False),
        ):
            for history, _ in app.query_collection(
                manifest["collection"],
                case["question"],
                [],
                use_rerank=True,
                think=False,
            ):
                pass
        if not captured:
            raise RuntimeError(history[-1]["content"])
        captured["retrieval_seconds"] = perf_counter() - started
        contexts[case["id"]] = captured
        print(case["id"], round(captured["retrieval_seconds"], 3), flush=True)
    assert collection.count() == before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contexts, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
