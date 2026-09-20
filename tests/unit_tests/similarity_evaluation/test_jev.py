from unittest.mock import Mock, patch

from gptcache.adapter.api import _get_eval
from gptcache.similarity_evaluation import JevEvaluation
from gptcache.similarity_evaluation.jev import JevEvaluation as JevEvaluationClass


def test_jev_evaluation():
    evaluation = JevEvaluation(api_key="test-key")
    assert evaluation.range() == (0.0, 1.0)

    mock_resp = Mock()
    mock_resp.json.return_value = {
        "answers": {k: {"noul": 0.9} for k in JevEvaluationClass.DIMENSIONS}
    }
    mock_resp.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        score = evaluation.evaluation(
            {"question": "What is the color of the sky?"},
            {"question": "What is the color of the sky?", "answer": "The sky is blue."},
        )

    assert 0.8 < score <= 1.0
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "jev-latest"
    for dim in JevEvaluationClass.DIMENSIONS:
        assert payload["questions"][dim]["type"] == "noul"
    assert "Cached request" in payload["state"]
    assert "Current request" in payload["state"]
    assert "Candidate answer" in payload["state"]
    assert "Answer date" in payload["state"]
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_jev_evaluation_min_score():
    evaluation = JevEvaluation(api_key="test-key")
    mock_resp = Mock()
    mock_resp.json.return_value = {
        "answers": {
            "task_identical": {"noul": 0.95},
            "context_matches": {"noul": 0.9},
            "format_ok": {"noul": 0.8},
            "is_fresh": {"noul": 0.4},  # fails -> min
            "no_missing_ctx": {"noul": 0.7},
        }
    }
    mock_resp.raise_for_status = Mock()
    with patch("requests.post", return_value=mock_resp):
        score = evaluation.evaluation(
            {"question": "q"}, {"question": "q", "answer": "a"}
        )
    assert score == 0.4


def test_jev_evaluation_error_returns_zero():
    evaluation = JevEvaluation(api_key="test-key")
    with patch("requests.post", side_effect=RuntimeError("boom")):
        score = evaluation.evaluation({"question": "q"}, {"answer": "a"})
    assert score == 0.0


def test_get_eval_jev():
    evaluation = _get_eval("jev", {"api_key": "test-key"})
    assert isinstance(evaluation, JevEvaluationClass)
    evaluation = _get_eval("typesafe", {"api_key": "test-key"})
    assert isinstance(evaluation, JevEvaluationClass)
