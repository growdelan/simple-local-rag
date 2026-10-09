"""Offline check that the evaluator retains production verification/retry semantics."""

import json
from unittest.mock import Mock
from oracle_eval import run_answer
from ui import app


def stream(text, reason="stop"):
    return iter(
        [
            {
                "message": {"role": "assistant", "content": text},
                "done": True,
                "done_reason": reason,
                "eval_count": 10,
                "eval_duration": 1000000,
            }
        ]
    )


def main():
    client = Mock()
    evidence = {
        "S1": {
            "text": "Biblioteka jest otwarta od ósmej.",
            "quote": "Biblioteka jest otwarta od ósmej.",
            "label": "sample.txt",
        }
    }
    context = "[S1] " + evidence["S1"]["text"]
    draft = json.dumps({"claims": [{"evidence_id": "S1", "answer": "Od ósmej."}]})
    client.chat.side_effect = [
        stream(draft),
        stream('{"assessment":"supported","accepted":true}'),
    ]
    result = run_answer(
        "test", "Kiedy biblioteka jest otwarta?", evidence, context, client
    )
    assert result["answer"] == "Od ósmej." and result["error"] is None
    assert [call["stage"] for call in result["calls"]] == ["draft", "verify_draft"]
    assert all(call.kwargs["think"] is False for call in client.chat.call_args_list)
    assert result["calls"][1]["num_predict"] == app.VERIFY_NUM_PREDICT

    client.chat.side_effect = [
        stream(draft),
        stream('{"assessment":"unsupported","accepted":false}'),
        stream("BRAK"),
    ]
    result = run_answer("test", "Godziny?", evidence, context, client)
    assert result["answer"] == app.NO_ANSWER
    assert result["draft_answer"] == "Od ósmej."
    assert [call["stage"] for call in result["calls"]] == [
        "draft",
        "verify_draft",
        "retry",
    ]

    client.chat.side_effect = [
        stream('{"claims":[]}'),
        stream("Od ósmej. [S1]"),
        stream('{"assessment":"supported","accepted":true}'),
    ]
    result = run_answer("test", "Godziny?", evidence, context, client)
    assert result["answer"] == "Od ósmej." and result["retry_answer"]
    assert [call["stage"] for call in result["calls"]] == [
        "draft",
        "retry",
        "verify_retry",
    ]

    client.chat.side_effect = [stream(draft), stream('{"accepted":true}', "length")]
    result = run_answer("test", "Godziny?", evidence, context, client)
    assert result["answer"] is None and result["error"].startswith(
        "GenerationLimitError:"
    )
    assert len(result["calls"]) == 2  # No unverified answer or hidden recovery.
    print(
        "PASS: evaluator draft, rejection, verified retry, token limit, stage timings and think=false"
    )


if __name__ == "__main__":
    main()
