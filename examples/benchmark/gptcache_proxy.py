"""OpenAI-Responses-compatible GPTCache proxy with JEV semantic matching.

POST /v1/responses (and /v1/chat/completions). On a new question it asks JEV
whether any cached answer correctly answers it; if JEV says yes (noul >= threshold)
it returns the cached answer (hit), otherwise it forwards upstream (miss) and caches.

Set JEV env var (TypeSafe API key) before starting.
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from datetime import datetime

import requests

from gptcache.similarity_evaluation.jev import reuse_questions

# Upstream model gateway the proxy forwards to on cache miss.
# Configure via env so the proxy can sit in front of any OpenAI-compatible backend.
UPSTREAM = os.getenv("GPTCACHE_UPSTREAM_URL", "https://api.openai.com/v1")
UPSTREAM_KEY = os.getenv("GPTCACHE_UPSTREAM_KEY", "")
MODEL = os.getenv("GPTCACHE_UPSTREAM_MODEL", "deepseek-v4-flash")
JEVI_URL = "https://api.typesafe.ai/v1/systemone"
JEVI_KEY = os.getenv("JEV")
THRESHOLD = float(os.getenv("GPTCACHE_THRESHOLD", "0.75"))
PORT = int(os.getenv("GPTCACHE_PORT", "8765"))

MEMORY = []  # list of {"question": ..., "answer": ..., "cached_at": iso-date}
STATS = {"hits": 0, "misses": 0}


def extract_question(body):
    inp = body.get("input", "")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        for m in reversed(inp):
            if isinstance(m, dict):
                c = m.get("content", "")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                if c:
                    return c
        return ""
    return str(inp)


def extract_answer(resp_json):
    for item in resp_json.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c["text"]
    return resp_json.get("text", "")


def forward_to_llm(body):
    clean = {
        "model": body.get("model", MODEL),
        "input": body.get("input"),
        "max_output_tokens": body.get("max_output_tokens", 500),
        "temperature": body.get("temperature", 0),
        "stream": False,
    }
    if body.get("instructions"):
        clean["instructions"] = body["instructions"]
    headers = {"Content-Type": "application/json"}
    if UPSTREAM_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_KEY}"
    resp = requests.post(
        UPSTREAM + "/responses",
        headers=headers,
        json=clean,
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"upstream {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def jev_judge(cached_question, question, answer, cached_at=None):
    now = datetime.now().strftime("%Y-%m-%d")
    state = (
        f"Cached request: {cached_question}\n"
        f"Current request: {question}\n"
        f"Candidate answer: {answer}\n"
        f"Answer date: {cached_at or 'unknown'}\n"
        f"Current date: {now}"
    )
    questions = reuse_questions(answer_date=cached_at, now=now)
    payload = {"state": state, "model": "jev-latest", "questions": questions}
    resp = requests.post(
        JEVI_URL,
        headers={"Authorization": f"Bearer {JEVI_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    answers = resp.json()["answers"]
    scores = {k: float(answers[k]["noul"]) for k in questions}
    return min(scores.values())


def semantic_search(question):
    best = None
    for entry in MEMORY:
        try:
            score = jev_judge(entry["question"], question, entry["answer"],
                              cached_at=entry["cached_at"])
        except Exception as e:  # pylint: disable=W0703
            print(f"[proxy] jev error: {e}", flush=True)
            continue
        if best is None or score > best[1]:
            best = (entry["answer"], score)
    return best


def build_cached_response(text, model):
    return {
        "id": "resp_gptcache",
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": [{"type": "message", "id": "msg_gptcache", "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "text": text,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "gptcache": True,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/stats"):
            data = json.dumps(STATS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.endswith("/memory"):
            data = json.dumps(MEMORY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.endswith("/models"):
            data = json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model",
                                                            "created": 0, "owned_by": "gptcache"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        # /seed: inject (question, answer, cached_at) to simulate a cache written at time T0
        if self.path.rstrip("/").endswith("/seed"):
            for item in body.get("pairs", []):
                q, a = item[0], item[1]
                cached_at = item[2] if len(item) > 2 else datetime.now().strftime("%Y-%m-%d")
                MEMORY.append({"question": q, "answer": a, "cached_at": cached_at})
            data = json.dumps({"seeded": len(body.get("pairs", []))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        model = body.get("model", MODEL)
        question = extract_question(body)

        best = semantic_search(question) if MEMORY else None
        if best and best[1] >= THRESHOLD:
            answer_text = best[0]
            STATS["hits"] += 1
            response_json = build_cached_response(answer_text, model)
            print(f"[proxy] HIT (jev={best[1]:.2f})", flush=True)
        else:
            try:
                result = forward_to_llm(body)
                answer_text = extract_answer(result)
                MEMORY.append({"question": question, "answer": answer_text,
                               "cached_at": datetime.now().strftime("%Y-%m-%d")})
                STATS["misses"] += 1
                response_json = result
                print(f"[proxy] MISS (jev={best[1] if best else None})", flush=True)
            except Exception as e:  # pylint: disable=W0703
                print(f"[proxy] upstream error: {e}", flush=True)
                response_json = build_cached_response(f"<error: {e}>", model)
                answer_text = f"<error: {e}>"

        if body.get("stream"):
            self._send_sse(answer_text, response_json)
        else:
            self._send_json(response_json)

    def _send_json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, text, response_json):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        rid = "resp_gptcache"
        iid = "msg_gptcache"
        item = {"type": "message", "id": iid, "status": "in_progress", "role": "assistant", "content": []}
        item_done = {"type": "message", "id": iid, "status": "completed", "role": "assistant",
                     "content": [{"type": "output_text", "text": text, "annotations": []}]}
        part = {"type": "output_text", "text": "", "annotations": []}
        part_done = {"type": "output_text", "text": text, "annotations": []}

        def emit(event, payload):
            self.wfile.write(f"event: {event}\n".encode())
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        emit("response.created", {"type": "response.created", "response": response_json})
        emit("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": item})
        emit("response.content_part.added", {"type": "response.content_part.added", "item_id": iid,
                                             "output_index": 0, "content_index": 0, "part": part})
        emit("response.output_text.delta", {"type": "response.output_text.delta", "item_id": iid,
                                            "output_index": 0, "content_index": 0, "delta": text})
        emit("response.output_text.done", {"type": "response.output_text.done", "item_id": iid,
                                           "output_index": 0, "content_index": 0, "text": text})
        emit("response.content_part.done", {"type": "response.content_part.done", "item_id": iid,
                                            "output_index": 0, "content_index": 0, "part": part_done})
        emit("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item_done})
        emit("response.completed", {"type": "response.completed", "response": response_json})


if __name__ == "__main__":
    print(f"GPTCache proxy (JEV semantic) listening on http://localhost:{PORT}/v1", flush=True)
    ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()
