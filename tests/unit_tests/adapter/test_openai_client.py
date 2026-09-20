import sys
import uuid
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gptcache import cache


def _install_fake_openai():
    fake = ModuleType("openai")
    fake.OpenAI = SimpleNamespace
    fake.AsyncOpenAI = SimpleNamespace
    sys.modules["openai"] = fake


_install_fake_openai()

with patch("gptcache.utils._check_library", return_value=True):
    from gptcache.adapter import openai_client
    from gptcache.adapter.openai_client import (
        get_message_from_openai_client_answer,
        get_stream_message_from_openai_client_answer,
    )


def _make_chat_completion(text, model="gpt-4o"):
    message = SimpleNamespace(role="assistant", content=text)
    choice = SimpleNamespace(index=0, finish_reason="stop", message=message)
    return SimpleNamespace(
        id="chatcmpl_test",
        choices=[choice],
        created=0,
        model=model,
        object="chat.completion",
    )


def test_normal_openai_client():
    cache.init()
    question = "calculate 1+3 " + uuid.uuid4().hex
    expect_answer = "the result is 4"

    fake_chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=Mock(return_value=_make_chat_completion(expect_answer))
        )
    )
    fake_client = SimpleNamespace(chat=fake_chat)
    openai_client.ChatCompletion.client = fake_client

    response = openai_client.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_openai_client_answer(response) == expect_answer

    response = openai_client.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_openai_client_answer(response) == expect_answer
    fake_chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_normal_openai_client_async():
    cache.init()
    question = "calculate 1+3 " + uuid.uuid4().hex
    expect_answer = "the result is 4"

    fake_chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=AsyncMock(return_value=_make_chat_completion(expect_answer))
        )
    )
    fake_client = SimpleNamespace(chat=fake_chat)
    openai_client.ChatCompletion.aclient = fake_client

    response = await openai_client.ChatCompletion.acreate(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_openai_client_answer(response) == expect_answer

    response = await openai_client.ChatCompletion.acreate(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_openai_client_answer(response) == expect_answer
    fake_chat.completions.create.assert_awaited_once()


def test_stream_openai_client():
    cache.init()
    question = "calculate 1+1 " + uuid.uuid4().hex
    expect_answer = "the result is 2"

    def _stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    delta=SimpleNamespace(role="assistant", content=expect_answer),
                )
            ]
        )

    fake_chat = SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=_stream())))
    fake_client = SimpleNamespace(chat=fake_chat)
    openai_client.ChatCompletion.client = fake_client

    response = openai_client.ChatCompletion.create(
        model="gpt-4o",
        stream=True,
        messages=[{"role": "user", "content": question}],
    )
    assert "".join(get_stream_message_from_openai_client_answer(c) for c in response) == expect_answer

    response = openai_client.ChatCompletion.create(
        model="gpt-4o",
        stream=True,
        messages=[{"role": "user", "content": question}],
    )
    assert "".join(get_stream_message_from_openai_client_answer(c) for c in response) == expect_answer
    fake_chat.completions.create.assert_called_once()
