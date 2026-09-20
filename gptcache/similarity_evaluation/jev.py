import os
from datetime import datetime
from typing import Any, Dict, Tuple

import requests

from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.utils.log import gptcache_log


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(item) for item in value)
    return str(value)


class JevEvaluation(SimilarityEvaluation):
    """Use TypeSafe's Jev model to decide whether a cached answer can be reused.

    Jev is TypeSafe AI's "System One" model that returns typed, probabilistic
    decisions instead of generated text. A cache entry can be reused only when
    *every* material condition holds, so we ask Jev a set of atomic ``noul``
    questions (task identity, context, format, freshness, missing context) and
    return the **minimum** score: a hit requires all dimensions to pass the
    ``similarity_threshold``.

    Reference: https://docs.typesafe.ai

    :param api_key: TypeSafe API key, defaults to ``TYPESAFE_API_KEY`` or ``JEV``.
    :type api_key: str
    :param model: Jev model id or alias, defaults to ``jev-latest``.
    :type model: str
    :param base_url: TypeSafe evaluation endpoint, defaults to ``https://api.typesafe.ai/v1/systemone``.
    :type base_url: str
    :param timeout: request timeout in seconds.
    :type timeout: float

    Example:
        .. code-block:: python

            from gptcache import cache, Config
            from gptcache.similarity_evaluation import JevEvaluation

            cache.init(
                similarity_evaluation=JevEvaluation(),
                config=Config(similarity_threshold=0.6),
            )
    """

    DIMENSIONS = (
        "task_identical",
        "context_matches",
        "format_ok",
        "is_fresh",
        "no_missing_ctx",
    )

    def __init__(
        self,
        api_key: str = None,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai/v1/systemone",
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.last_scores: Dict[str, float] = {}

    def _questions(self, answer_date: str = None, now: str = None) -> Dict[str, Any]:
        criteria = {
            "true": "This condition is satisfied.",
            "false": "This condition is violated or cannot be established.",
        }
        if answer_date and now:
            is_fresh_instruction = (
                f"The answer to this question remains unchanged between {answer_date} "
                f"(the answer date) and {now} (the current date)."
            )
        else:
            is_fresh_instruction = (
                "The answer to this question remains unchanged up to the current date."
            )
        return {
            "task_identical": {
                "type": "noul",
                "criteria": criteria,
                "instructions": "The task described by the cached request is completely identical "
                               "to the task described by the current request — they ask for the "
                               "exact same operation to be performed.",
            },
            "context_matches": {
                "type": "noul",
                "criteria": criteria,
                "instructions": "The cached request and the current request refer to the same "
                               "entities, subjects, numbers, dates, and constraints. References "
                               "such as 'this', 'it', and 'the above' resolve correctly.",
            },
            "format_ok": {
                "type": "noul",
                "criteria": criteria,
                "instructions": "The candidate answer satisfies the required language, format, "
                               "detail level, and tone.",
            },
            "is_fresh": {
                "type": "noul",
                "criteria": criteria,
                "instructions": is_fresh_instruction,
            },
            "no_missing_ctx": {
                "type": "noul",
                "criteria": criteria,
                "instructions": "Returning the candidate answer does not depend on unavailable "
                               "prior context, another user's private information, missing "
                               "attachments, or unsupported assumptions.",
            },
        }

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **_
    ) -> float:
        """Evaluate whether the cached answer can be reused for the incoming question.

        :param src_dict: the user request params, expected to contain ``question``.
        :param cache_dict: the cache request params, expected to contain ``question``,
            ``answer`` and optionally ``cache_data`` (with ``create_on``).
        :return: minimum score across all dimensions (0-1). A hit requires every
            dimension to exceed ``similarity_threshold``.
        """
        question = _stringify(src_dict.get("question"))
        cached_question = _stringify(cache_dict.get("question"))
        answer = _stringify(cache_dict.get("answer", cache_dict.get("question")))

        cache_data = cache_dict.get("cache_data")
        answer_date = None
        if cache_data is not None and getattr(cache_data, "create_on", None):
            answer_date = cache_data.create_on.strftime("%Y-%m-%d")

        now = datetime.now().strftime("%Y-%m-%d")
        state = (
            f"Cached request: {cached_question}\n"
            f"Current request: {question}\n"
            f"Candidate answer: {answer}\n"
            f"Answer date: {answer_date or 'unknown'}\n"
            f"Current date: {now}"
        )

        payload = {
            "state": state,
            "model": self.model,
            "questions": self._questions(answer_date=answer_date, now=now),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            scores = {
                key: float(data["answers"][key]["noul"]) for key in self.DIMENSIONS
            }
            self.last_scores = scores
            return min(scores.values())
        except Exception as e:  # pylint: disable=W0703
            gptcache_log.warning("failed to evaluate with Jev, error: %s", e)
            return 0.0

    def range(self) -> Tuple[float, float]:
        return 0.0, 1.0
