"""Frozen positive/negative claims: assess verification independently of drafting."""

import argparse
import hashlib
import json
from pathlib import Path
from oracle_eval import MeteredClient, load_oracles
from ui import app
import ollama

# Literal, controlled contradictions and missing requested facts, not generated labels.
PAIRS = [
    (
        "księżyce",
        "Na niebie były dwa księżyce: duży żółty oraz mały zielonkawy i nieregularny.",
        "Mały księżyc był żółty, a duży zielonkawy.",
    ),
    (
        "przejście",
        "Aomame zeszła schodami awaryjnymi ze stołecznej autostrady.",
        "Aomame przeniosła się do 1Q84, bo krajobraz przypominał jej obcy las.",
    ),
    (
        "miejsce urodzenia",
        "Tamaru urodził się na Sachalinie.",
        "Tamaru urodził się na Hokkaido.",
    ),
    ("pseudonim", "Fukaeri.", "Występowała jako autorka Powietrznej poczwarki."),
    ("praca Tengo", "Tengo uczył matematyki.", "Tengo uczył historii."),
    ("kompozytor", "Janáček.", "Dvořák."),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    manifest = json.loads(Path(__file__).with_name("oracle_cases.json").read_text())
    contexts = load_oracles(manifest, Path("data/1Q84_full"))
    cases = {case["id"]: case for case in manifest["cases"]}
    client = ollama.Client(host=app.OLLAMA_BASE_URL, timeout=app.RAG_TIMEOUT)
    digest = next(m.digest for m in client.list().models if m.model == args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case_id, positive, negative in PAIRS:
        for expected, answer in ((True, positive), (False, negative)):
            meter = MeteredClient(client)
            meter.stage = "verifier_challenge"
            llm = app._make_answer_llm(args.model, think=False, client=meter)
            evidence, _ = contexts[case_id]
            source_id = "S2" if case_id == "księżyce" else "S1"
            claim = {"evidence_id": source_id, "answer": answer}
            error = None
            try:
                accepted = bool(
                    app._verify_grounding(
                        llm,
                        json.dumps({"claims": [claim]}),
                        evidence,
                        cases[case_id]["question"],
                    )
                )
            except Exception as exc:
                accepted = None
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "id": case_id,
                "model": args.model,
                "model_digest": digest,
                "think": False,
                "source_id": source_id,
                "source_sha256": hashlib.sha256(
                    evidence[source_id]["text"].encode()
                ).hexdigest(),
                "claim": answer,
                "expected_accept": expected,
                "accepted": accepted,
                "error": error,
                "calls": meter.calls,
            }
            with args.output.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(case_id, expected, accepted, error, flush=True)


if __name__ == "__main__":
    main()
