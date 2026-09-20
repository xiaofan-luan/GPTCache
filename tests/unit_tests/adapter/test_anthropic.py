import sys
import uuid
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gptcache import cache
from gptcache.utils.response import get_message_from_anthropic_answer


class _FakeAPIError(Exception):
    pass


def _install_fake_anthropic():
    fake = ModuleType("anthropic")
    fake.APIError = _FakeAPIError
    fake.Anthropic = SimpleNamespace
    fake.AsyncAnthropic = SimpleNamespace
    sys.modules["anthropic"] = fake


_install_fake_anthropic()

with patch("gptcache.utils._check_library", return_value=True):
    from gptcache.adapter import anthropic as cache_anthropic


def _make_message(text):
    return SimpleNamespace(
        id="msg_test",
        content=[SimpleNamespace(type="text", text=text)],
        role="assistant",
        model="claude-3-5-sonnet-20240620",
        stop_reason="end_turn",
    )


def test_normal_anthropic():
    cache.init()
    question = "calculate 1+3 " + uuid.uuid4().hex
    expect_answer = "the result is 4"

    fake_messages = SimpleNamespace(create=Mock(return_value=_make_message(expect_answer)))
    fake_client = SimpleNamespace(messages=fake_messages)
    cache_anthropic.ChatCompletion.client = fake_client

    response = cache_anthropic.ChatCompletion.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_anthropic_answer(response) == expect_answer

    response = cache_anthropic.ChatCompletion.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_anthropic_answer(response) == expect_answer
    fake_messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_normal_anthropic_async():
    cache.init()
    question = "calculate 1+3 " + uuid.uuid4().hex
    expect_answer = "the result is 4"

    fake_messages = SimpleNamespace(create=AsyncMock(return_value=_make_message(expect_answer)))
    fake_client = SimpleNamespace(messages=fake_messages)
    cache_anthropic.ChatCompletion.aclient = fake_client

    response = await cache_anthropic.ChatCompletion.acreate(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_anthropic_answer(response) == expect_answer

    response = await cache_anthropic.ChatCompletion.acreate(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_anthropic_answer(response) == expect_answer
    fake_messages.create.assert_awaited_once()


def test_stream_anthropic():
    cache.init()
    question = "calculate 1+1 " + uuid.uuid4().hex
    expect_answer = "the result is 2"

    def _stream():
        yield SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=expect_answer))

    fake_messages = SimpleNamespace(create=Mock(return_value=_stream()))
    fake_client = SimpleNamespace(messages=fake_messages)
    cache_anthropic.ChatCompletion.client = fake_client

    response = cache_anthropic.ChatCompletion.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        stream=True,
        messages=[{"role": "user", "content": question}],
    )
    chunks = list(response)
    assert chunks[0].delta.text == expect_answer

    response = cache_anthropic.ChatCompletion.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        stream=True,
        messages=[{"role": "user", "content": question}],
    )
    cached_chunks = list(response)
    assert cached_chunks[0].delta.text == expect_answer
    fake_messages.create.assert_called_once()
