import json
import os
import queue
import re
import subprocess
import threading
import time

import httpx
from pydantic import field_validator

from app.core import speech
from app.core.settings import AppSettings, strip_url

DEFAULT_CHAT_SYSTEM = (
    "You are a friendly Australian helper for kids. Speak simply and warmly, "
    "in one or two short sentences."
)


class DialogueSettings(AppSettings):
    llm_url: str = ""
    llm_model: str = "Qwen/Qwen3.5-4B"
    chat_max_tokens: int = 256
    chat_system: str = DEFAULT_CHAT_SYSTEM

    @field_validator("llm_url", mode="before")
    @classmethod
    def _strip_url(cls, value: str | None) -> str:
        return strip_url(value)

    @property
    def llm_chat_url(self) -> str:
        if not self.llm_url:
            return ""
        return f"{self.llm_url}/v1/chat/completions"


dialogue_settings = DialogueSettings()

LLM_URL = dialogue_settings.llm_url
LLM_CHAT_URL = dialogue_settings.llm_chat_url
LLM_MODEL = dialogue_settings.llm_model
CHAT_MAX_TOKENS = dialogue_settings.chat_max_tokens
SYSTEM = dialogue_settings.chat_system
PLAY = speech.PLAY
TTS_ON = bool(speech.TTS_URL)

LLM_CLIENT = httpx.Client(timeout=600, limits=httpx.Limits(max_keepalive_connections=4))


class ClauseStreamer:
    def __init__(self, first_min=15, soft_cap=70, hard_cap=140):
        self.buf = ""
        self.first_done = False
        self.first_min = first_min
        self.soft_cap = soft_cap
        self.hard_cap = hard_cap

    def _term_cut(self):
        for m in re.finditer(r"[.!?]+(?=\s)", self.buf):
            i = m.end()
            toks = self.buf[:i].split()
            tail = re.sub(r"[.!?]+$", "", toks[-1]).rstrip(".").lower() if toks else ""
            if tail in speech.ABBREV:
                continue
            return i
        return None

    def _soft_cut(self, min_start):
        for m in re.finditer(r"[,;:](?=\s)", self.buf):
            if m.start() >= min_start:
                return m.end()
        return None

    def _find_cut(self):
        t = self._term_cut()
        if not self.first_done:
            c = self._soft_cut(self.first_min)
            cands = [x for x in (t, c) if x is not None]
            if cands:
                return min(cands)
        else:
            if t is not None:
                return t
            if len(self.buf) >= self.soft_cap:
                c = self._soft_cut(0)
                if c is not None:
                    return c
        if len(self.buf) >= self.hard_cap:
            sp = self.buf.rfind(" ", 0, self.hard_cap)
            if sp > 0:
                return sp + 1
        return None

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            clause = self.buf[:cut].strip()
            self.buf = self.buf[cut:].lstrip()
            if clause:
                out.append(clause)
                self.first_done = True
        return out

    def flush(self) -> list[str]:
        c = self.buf.strip()
        self.buf = ""
        if c:
            self.first_done = True
            return [c]
        return []


def llm_payload(messages: list[dict]) -> dict:
    return {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": CHAT_MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }


def parse_sse_delta(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)["choices"][0].get("delta", {}).get("content") or None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def llm_stream(messages):
    with LLM_CLIENT.stream("POST", LLM_CHAT_URL, json=llm_payload(messages)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if delta := parse_sse_delta(line):
                yield delta


def converse(messages) -> str:
    if not LLM_URL:
        raise SystemExit("set LLM_URL to the Qwen3.5-4B endpoint (see llm/qwen3_5_4b.py)")

    clause_q: queue.Queue = queue.Queue()
    t0 = time.time()
    st = {"ttft": None}

    def llm_reader():
        streamer = ClauseStreamer()
        try:
            for delta in llm_stream(messages):
                if st["ttft"] is None and delta.strip():
                    st["ttft"] = time.time() - t0
                for clause in streamer.feed(delta):
                    clause_q.put(clause)
            for clause in streamer.flush():
                clause_q.put(clause)
        except Exception as e:
            clause_q.put(("__ERR__", e))
        clause_q.put(None)

    threading.Thread(target=llm_reader, daemon=True).start()

    if not TTS_ON:
        parts = []
        while True:
            item = clause_q.get()
            if item is None:
                break
            if isinstance(item, tuple):
                print(f"  [llm error] {item[1]}")
                continue
            if not parts:
                print(f"  >> LLM TTFT {st['ttft']:.2f}s")
            parts.append(item)
            print(f"  [{len(parts) - 1}] {item}")
        print(f"  -- reply in {time.time() - t0:.2f}s ({len(parts)} clauses); set TTS_URL to speak it")
        return " ".join(parts)

    audio_q: queue.Queue = queue.Queue()

    def synth_worker():
        i = 0
        while True:
            item = clause_q.get()
            if item is None:
                break
            if isinstance(item, tuple):
                audio_q.put(item)
                continue
            try:
                tt, b = speech.synth(item)
                audio_q.put((i, item, b, tt))
                i += 1
            except Exception as e:
                audio_q.put(("__ERR__", e))
        audio_q.put(None)

    threading.Thread(target=synth_worker, daemon=True).start()

    os.makedirs(speech.OUT_DIR, exist_ok=True)
    first_audio = None
    collected, parts, total_audio = [], [], 0.0
    while True:
        item = audio_q.get()
        if item is None:
            break
        if isinstance(item, tuple) and item[0] == "__ERR__":
            print(f"  [error] {item[1]}")
            continue
        i, text, b, tt = item
        if first_audio is None:
            first_audio = time.time() - t0
            print(f"  >> first audio {first_audio:.2f}s  (LLM TTFT {st['ttft']:.2f}s)")
        parts.append(text)
        path = os.path.join(speech.OUT_DIR, f"chat_{i:02d}.wav")
        with open(path, "wb") as f:
            f.write(b)
        collected.append((i, b))
        dur = speech.wav_dur(b)
        total_audio += dur
        print(f"  [{i}] synth {tt:5.2f}s  audio {dur:5.2f}s  | {text[:55]!r}")
        if PLAY:
            subprocess.run(["afplay", path], check=False)

    wall = time.time() - t0
    print(f"  -- turn wall {wall:.2f}s for {total_audio:.2f}s audio ({len(parts)} clauses)")
    if collected:
        stitched = speech.stitch([b for _, b in sorted(collected)])
        sp = os.path.join(speech.OUT_DIR, "chat.wav")
        with open(sp, "wb") as f:
            f.write(stitched)
        print(f"  -- stitched -> {sp} ({speech.wav_dur(stitched):.2f}s)")
    return " ".join(parts)


__all__ = [
    "ClauseStreamer",
    "CHAT_MAX_TOKENS",
    "LLM_CHAT_URL",
    "LLM_CLIENT",
    "LLM_MODEL",
    "LLM_URL",
    "PLAY",
    "SYSTEM",
    "TTS_ON",
    "converse",
    "llm_payload",
    "llm_stream",
    "parse_sse_delta",
]
