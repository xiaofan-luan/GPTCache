from datetime import datetime
from types import SimpleNamespace
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


def test_documented_similarity_threshold_is_balanced_default():
    assert "similarity_threshold=0.70" in JevEvaluationClass.__doc__


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


def test_freshness_receives_cache_date_and_has_separate_criteria():
    evaluation = JevEvaluation(api_key="test-key")
    response = Mock()
    response.json.return_value = {
        "answers": {key: {"noul": 0.9} for key in evaluation.DIMENSIONS}
    }
    with patch("requests.post", return_value=response) as post:
        evaluation.evaluation(
            {"question": "q"},
            {"question": "q", "answer": "a",
             "cache_data": SimpleNamespace(create_on=datetime(2026, 9, 1))},
        )
    payload = post.call_args.kwargs["json"]
    assert "2026-09-01" in payload["questions"]["is_fresh"]["instructions"]
    assert payload["questions"]["is_fresh"]["criteria"] != payload["questions"]["format_ok"]["criteria"]
    assert "Answer date: 2026-09-01" in payload["state"]


def test_reuse_questions_exclude_answer_correctness_and_define_format_boundary():
    questions = JevEvaluation(api_key="test-key")._questions()
    for name in ["task_identical", "context_matches", "format_ok", "no_missing_ctx"]:
        assert "Answer correctness is outside" in questions[name]["instructions"]
    format_prompt = questions["format_ok"]["instructions"]
    assert "one paragraph versus multiple paragraphs" in format_prompt
    assert "mid-sentence beginning or ending" in format_prompt
    assert "TODO, mock, echo" in format_prompt
    assert "Only when the candidate itself begins" in format_prompt
    assert "do not concatenate it to the request" in format_prompt
    assert "user-fillable fields" in format_prompt
    assert "case-sensitive tokens" in format_prompt
    assert "clipped optional elaboration" in format_prompt
    assert "do not invent one" in format_prompt
    task_prompt = questions["task_identical"]["instructions"]
    assert "Allow paraphrases" in task_prompt
    assert "same material operation and deliverable" in task_prompt
    assert "real objective-strength change" in task_prompt
    assert "'ideal', 'best', and 'most effective'" in task_prompt
    context_prompt = questions["context_matches"]["instructions"]
    assert "exact matching semantics" in context_prompt
    assert "geography, jurisdiction, organization, and requested source" in context_prompt
    missing_prompt = questions["no_missing_ctx"]["instructions"]
    assert "truncated inside an unfinished clause" in missing_prompt
    assert "Require visible evidence" in missing_prompt
    assert "format_ok checks" in missing_prompt
    assert "Do not use this condition to judge candidate-answer formatting" in missing_prompt
