"""Opcjonalna regresja na istniejącej kolekcji 1Q84; nie zmienia dokumentów."""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="1Q84_full")
    parser.add_argument("--model", default=app.STANDARD_MODEL)
    args = parser.parse_args()
    collection = app._get_chroma_client().get_collection(args.collection)
    initial_count = collection.count()
    cases = [
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
            lambda text: text.startswith("Nie znaleziono odpowiedzi"),
        ),
    ]
    failures = []
    for name, question, check in cases:
        started = perf_counter()
        for history, _ in app.query_collection(
            args.collection, question, [], use_rerank=True, model_name=args.model
        ):
            pass
        answer = history[-1]["content"]
        # Nie zaliczamy testu tylko dlatego, że słowo wystąpiło w cytacie.
        assertions = "\n".join(
            part.split("\n\nUzasadnienie: ")[0] for part in answer.split("\n\n---\n\n")
        )
        success = check(assertions)
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
