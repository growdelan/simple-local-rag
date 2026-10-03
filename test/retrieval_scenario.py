"""Regresja: fakt na końcu długiego źródła nie może zniknąć w rerankingu."""

import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import app
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle


class WordTokenizer:
    def __call__(self, text, **kwargs):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}

    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2


class TruncatingEncoder:
    tokenizer = WordTokenizer()

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        result = []
        for query, text in pairs:
            budget = app.RERANK_MAX_LENGTH - len(query.split()) - 4
            visible = text.split()[:budget]
            result.append(
                10 if "Rynku." in visible else 2 if "dworzec" in visible else 0
            )
        return result


def main():
    source = (
        " ".join(["Zapis neutralnej obserwacji."] * 15)
        + " Lena spotkała inspektora na Rynku."
    )
    correct = NodeWithScore(
        node=TextNode(text=source, metadata={"file_name": "scenariusz.txt"})
    )
    distractor = NodeWithScore(node=TextNode(text="Lena minęła dworzec po spotkaniu."))
    reranker = object.__new__(app.LocalReranker)
    reranker.model = TruncatingEncoder()
    query = QueryBundle(query_str="Gdzie spotkanie?")
    with (
        patch.object(app, "RERANK_MAX_LENGTH", 16),
        patch.object(app, "RERANK_WINDOW_OVERLAP", 3),
        patch.object(app, "RERANK_TOP_N", 1),
    ):
        with patch.object(app, "RERANK_WINDOW_WEIGHT", 0):
            assert (
                reranker.postprocess_nodes([correct, distractor], query)[0]
                is distractor
            )
        with patch.object(app, "RERANK_WINDOW_WEIGHT", 0.5):
            selected = reranker.postprocess_nodes([correct, distractor], query)
            assert selected == [correct]
            assert correct.node.text == source
            evidence, _ = app._prepare_evidence(selected)
            assert evidence["S1"]["quote"] == source
            assert evidence["S1"]["label"] == "scenariusz.txt"
            assert all(q == query.query_str for q, _ in reranker.model.pairs)
            assert all(len(t.split()) <= 10 for _, t in reranker.model.pairs)
            assert reranker.postprocess_nodes([], query) == []
    tokenizer = WordTokenizer()
    assert app._rerank_passages("", tokenizer, "Pytanie?", 16, 3) == [""]
    assert app._rerank_passages("Krótki tekst.", tokenizer, "Pytanie?", 16, 3) == [
        "Krótki tekst."
    ]
    assert app._rerank_passages(source, tokenizer, "a " * 20, 16, 3) == [source]
    assert app._rerank_passages(source, tokenizer, "a " * 8, 16, 3) == [source]
    for overlap in (0, 3, 100):
        windows = app._rerank_passages(source, tokenizer, "Pytanie?", 16, overlap)
        assert windows[0].startswith("Zapis") and windows[-1].endswith("na Rynku.")
        assert all(window in source for window in windows)
    # Identyczny zbiór źródeł zachowuje kolejność i identyfikatory S1, S2.
    alpha = NodeWithScore(node=TextNode(text="alpha " * 30))
    beta = NodeWithScore(node=TextNode(text="beta " * 30))

    def scores(pairs, **kwargs):
        return [
            (
                (4 if "alpha" in text else 3)
                if len(text.split()) > 10
                else (0 if "alpha" in text else 10)
            )
            for _, text in pairs
        ]

    with (
        patch.object(app, "RERANK_WINDOW_WEIGHT", 0.5),
        patch.object(app, "RERANK_WINDOW_TOKENS", 10),
        patch.object(app, "RERANK_TOP_N", 2),
        patch.object(reranker.model, "predict", side_effect=scores),
    ):
        assert reranker.postprocess_nodes([alpha, beta], query) == [alpha, beta]
        assert beta.score > alpha.score
    print(
        "PASS: informacja na końcu źródła, limit pary, krótki dokument, cytaty i brak duplikatów."
    )


if __name__ == "__main__":
    main()
