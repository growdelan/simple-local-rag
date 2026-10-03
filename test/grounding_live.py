"""Opcjonalna regresja na istniejącej kolekcji 1Q84; nie zmienia dokumentów."""

import argparse
import logging
import json
import sys
import unicodedata
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="1Q84_full")
    parser.add_argument("--model", default=app.STANDARD_MODEL)
    parser.add_argument("--think", choices=["default", "on", "off"], default="default")
    parser.add_argument(
        "--suite", choices=["core", "holdout", "fresh", "all"], default="core"
    )
    parser.add_argument("--rerank", choices=["on", "off"], default="on")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--output", help="Opcjonalny plik JSONL z odpowiedziami bez cytatów"
    )
    args = parser.parse_args()
    collection = app._get_chroma_client().get_collection(args.collection)
    initial_count = collection.count()
    cases = [
        (
            "miejsce zabójstwa",
            "Gdzie Aomame zabiła lidera?",
            lambda text: "okura" in text.lower() and "hotel" in text.lower(),
        ),
        (
            "miejsce zabójstwa — parafraza",
            "W jakim miejscu Aomame zabiła Lidera?",
            lambda text: "okura" in text.lower() and "hotel" in text.lower(),
        ),
        (
            "detektyw",
            "Jak nazywał się mały detektyw który ścigał Aomame po zabiciu lidera?",
            lambda text: "ushikawa" in text.lower() and "rekrut" not in text.lower(),
        ),
        (
            "księżyce",
            "Co niezwykłego Aomame zauważa na niebie w świecie 1Q84? Jak wyglądają widoczne tam księżyce?",
            lambda text: all(
                part in text.lower() for part in ("księży", "żół", "ziel")
            ),
        ),
        (
            "przejście",
            "W jaki sposób Aomame dostała się do świata 1Q84?",
            lambda text: "schod" in text.lower() and "autostrad" in text.lower(),
        ),
        (
            "brak danych",
            "Jaki jest dokładny numer konta bankowego Aomame? Podaj wszystkie cyfry.",
            lambda text: text == app.NO_ANSWER,
        ),
    ]

    def normalized(text):
        return "".join(
            char
            for char in unicodedata.normalize("NFKD", text.lower())
            if not unicodedata.combining(char)
        )

    holdout = [
        (
            "kompozytor",
            "Kto skomponował Sinfoniettę, której Aomame słuchała w taksówce?",
            lambda text: "janac" in normalized(text),
        ),
        (
            "pseudonim",
            "Pod jakim pseudonimem Eriko Fukada występowała jako autorka Powietrznej poczwarki?",
            lambda text: "fukaeri" in text.lower(),
        ),
        (
            "praca Tengo",
            "Jakiego przedmiotu uczył Tengo?",
            lambda text: "matematyk" in text.lower(),
        ),
        (
            "ochroniarz",
            "Jak nazywał się ochroniarz starszej pani?",
            lambda text: "tamaru" in text.lower(),
        ),
        (
            "policjantka",
            "Jak nazywała się policjantka, z którą zaprzyjaźniła się Aomame?",
            lambda text: "ayumi" in text.lower(),
        ),
    ]
    fresh = [
        (
            "litera Q",
            "Co według Aomame oznacza litera Q w nazwie 1Q84?",
            lambda text: "zapytania" in text.lower() or "question mark" in text.lower(),
        ),
        (
            "dziewczynka",
            "Jak nazywała się dziewczynka, która trafiła do starszej pani po ucieczce z Sakigake?",
            lambda text: "tsubasa" in text.lower(),
        ),
        (
            "miejsce urodzenia",
            "Na jakiej wyspie urodził się Tamaru?",
            lambda text: "sachalin" in text.lower(),
        ),
    ]
    if args.suite == "holdout":
        cases = holdout
    elif args.suite == "fresh":
        cases = fresh
    elif args.suite == "all":
        cases += holdout + fresh
    if args.repeat < 1:
        parser.error("--repeat musi być dodatnie")
    failures = []
    for run, (name, question, check) in (
        (run, case) for run in range(1, args.repeat + 1) for case in cases
    ):
        started = perf_counter()
        for history, _ in app.query_collection(
            args.collection,
            question,
            [],
            use_rerank=args.rerank == "on",
            model_name=args.model,
            think=None if args.think == "default" else args.think == "on",
        ):
            pass
        answer = history[-1]["content"]
        # Nie zaliczamy testu tylko dlatego, że słowo wystąpiło w cytacie.
        assertions = "\n".join(
            part.split("\n\nUzasadnienie: ")[0] for part in answer.split("\n\n---\n\n")
        )
        success = check(assertions)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "id": name,
                            "question": question,
                            "answer": assertions,
                            "passed": success,
                            "check_type": "keyword_heuristic_requires_manual_review",
                            "run": run,
                            "suite": args.suite,
                            "rerank": args.rerank,
                            "rerank_model": app.RERANK_MODEL_NAME,
                            "window_weight": app.RERANK_WINDOW_WEIGHT,
                            "seconds": perf_counter() - started,
                            "think": args.think,
                            "model": args.model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(
            f'{name}: {"PASS" if success else "FAIL"}, {perf_counter() - started:.2f}s\n{answer}\n',
            flush=True,
        )
        if not success:
            failures.append(name)
    assert collection.count() == initial_count, "Zmieniona liczba nodów"
    assert not failures, failures
    print("PASS: wszystkie próby; heurystyki tekstowe nie zastępują oceny cytatów.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
