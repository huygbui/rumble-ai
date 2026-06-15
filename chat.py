# chat.py
# End-to-end voice-loop scaffold: stream the dialogue LLM (Qwen3.5-4B) and OVERLAP TTS
# synthesis with generation, so the FIRST clause is spoken while the rest of the reply is
# still being generated. Felt turn-latency tracks (LLM TTFT + first-clause gen + first-clause
# synth) -- NOT the whole reply. This is the responsiveness win from docs/omnivoice-bench.md
# (OmniVoice can't stream within a clause -- NAR diffusion -- so we get low latency by
# streaming the *text* and chunking it instead).
#
# Pipeline (3 stages, fully overlapped):
#   LLM SSE stream --(tokens)--> ClauseStreamer --(clauses)--> say.synth --(wav)--> afplay
#   thread: llm_reader            (in llm_reader)   thread: synth_worker   main: player
# While clause 0 PLAYS, clause 1 SYNTHESIZES, and the LLM is still GENERATING clause 2.
#
#   export LLM_URL="https://<workspace>--qwen3-5-4b-llm-serve.modal.run"
#   export TTS_URL="https://<workspace>--omnivoice-tts-serve.modal.run"   # omit -> text-only
#   echo "tell me a tiny story about a wombat" | python chat.py
#   CHAT_TEXT="g'day!" python chat.py     # one turn via env
#   PLAY=1 ...                            # speak through afplay as clauses arrive (macOS)
#   CHAT_REPL=1 ...                       # interactive multi-turn loop (keeps history)
#
# NOTE: STT (the mic -> text front-end) is a separate, not-yet-built stage (see stt/). Input
# here is TYPED text standing in for the eventual transcript. The kid-safety guardrail
# (input + output moderation) is ALSO a separate layer -- this scaffold is dialogue+voice only.
# Writes per-clause wavs to ./out (chat_NN.wav) + a stitched ./out/chat.wav, and prints the
# two metrics that matter: LLM TTFT and time-to-first-audio.
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import requests

import say  # reuse synth(), stitch(), wav_dur(), OUT_DIR, ABBREV, the pooled TTS Session, etc.

# --- LLM endpoint (plain vLLM OpenAI-compatible; see llm/qwen3_5_4b.py) ----------------
LLM_BASE = os.environ.get("LLM_URL", "").rstrip("/")
LLM_CHAT_URL = f"{LLM_BASE}/v1/chat/completions"
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-4B")
MAX_TOKENS = int(os.environ.get("CHAT_MAX_TOKENS", "256"))
# Persona / register go in the system prompt (kid-safety does NOT -- that's the guardrail layer).
SYSTEM = os.environ.get(
    "CHAT_SYSTEM",
    "You are a friendly Australian helper for kids. Speak simply and warmly, "
    "in one or two short sentences.",
)
PLAY = os.environ.get("PLAY") not in (None, "", "0")
TTS_ON = bool(say.BASE)  # if TTS_URL is unset, run text-only (stream + print, no synth/GPU)

# One pooled keep-alive connection to the LLM (separate host from TTS -> its own Session).
LLM_SESSION = requests.Session()
LLM_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=4, max_retries=1))
LLM_SESSION.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=4, max_retries=1))


# --- Incremental clause splitter -------------------------------------------------------
# split_clauses (say.py) works on a COMPLETE reply; streaming needs to emit a clause the
# instant a boundary arrives. Same decimal/abbrev safety (reuses say.ABBREV): only cut on a
# terminator FOLLOWED BY whitespace (so "0.22.1" / "3.5" never shatter), skipping known
# abbreviations ("Dr. Smith"). The FIRST clause is cut as early as possible (terminator OR a
# comma past ~15 chars) -> fast first audio; later clauses flush whole sentences, falling back
# to a comma when a sentence runs long, and to a word boundary only past a hard cap.
class ClauseStreamer:
    def __init__(self, first_min=15, soft_cap=70, hard_cap=140):
        self.buf = ""
        self.first_done = False
        self.first_min = first_min
        self.soft_cap = soft_cap
        self.hard_cap = hard_cap

    def _term_cut(self):
        # earliest sentence terminator followed by whitespace, not an abbreviation
        for m in re.finditer(r"[.!?]+(?=\s)", self.buf):
            i = m.end()
            toks = self.buf[:i].split()
            tail = re.sub(r"[.!?]+$", "", toks[-1]).rstrip(".").lower() if toks else ""
            if tail in say.ABBREV:
                continue
            return i
        return None

    def _soft_cut(self, min_start):
        # earliest comma/semicolon/colon (followed by whitespace) starting at >= min_start
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
        if len(self.buf) >= self.hard_cap:  # never let a clause grow unbounded
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


# --- LLM streaming ---------------------------------------------------------------------
def llm_stream(messages):
    # Yields content deltas from the OpenAI-compatible SSE stream. enable_thinking=false +
    # Qwen3.5's recommended non-thinking sampling (from llm/qwen3_5_4b.py) -> fast, no CoT.
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }
    r = LLM_SESSION.post(LLM_CHAT_URL, json=payload, stream=True, timeout=600)
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0].get("delta", {})
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        chunk = delta.get("content") or ""
        if chunk:
            yield chunk


# --- One conversational turn -----------------------------------------------------------
def converse(messages) -> str:
    # Returns the assistant's full reply text. Streams it; if TTS is on, synthesizes each
    # clause as it arrives and plays in order while later clauses synth and the LLM generates.
    if not LLM_BASE:
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
        except Exception as e:  # surface LLM/transport errors to the consumer, don't hang
            clause_q.put(("__ERR__", e))
        clause_q.put(None)

    threading.Thread(target=llm_reader, daemon=True).start()

    # --- text-only mode: no TTS_URL -> just stream + print clauses (dev w/o GPU) ---
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

    # --- voice mode: synth_worker pulls clauses, player consumes ready audio in order ---
    audio_q: queue.Queue = queue.Queue()

    def synth_worker():
        i = 0
        while True:
            item = clause_q.get()
            if item is None:
                break
            if isinstance(item, tuple):  # ("__ERR__", exc)
                audio_q.put(item)
                continue
            try:
                tt, b = say.synth(item)
                audio_q.put((i, item, b, tt))
                i += 1
            except Exception as e:
                audio_q.put(("__ERR__", e))
        audio_q.put(None)

    threading.Thread(target=synth_worker, daemon=True).start()

    os.makedirs(say.OUT_DIR, exist_ok=True)
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
        path = os.path.join(say.OUT_DIR, f"chat_{i:02d}.wav")
        with open(path, "wb") as f:
            f.write(b)
        collected.append((i, b))
        dur = say.wav_dur(b)
        total_audio += dur
        print(f"  [{i}] synth {tt:5.2f}s  audio {dur:5.2f}s  | {text[:55]!r}")
        if PLAY:
            subprocess.run(["afplay", path], check=False)

    wall = time.time() - t0
    print(f"  -- turn wall {wall:.2f}s for {total_audio:.2f}s audio ({len(parts)} clauses)")
    if collected:  # one seamless artifact for the whole reply (trim/fade/one-pause joins)
        stitched = say.stitch([b for _, b in sorted(collected)])
        sp = os.path.join(say.OUT_DIR, "chat.wav")
        with open(sp, "wb") as f:
            f.write(stitched)
        print(f"  -- stitched -> {sp} ({say.wav_dur(stitched):.2f}s)")
    return " ".join(parts)


def main() -> None:
    print(f"LLM={LLM_CHAT_URL or '(unset)'}  model={LLM_MODEL}")
    print(f"TTS={'on -> ' + say.URL if TTS_ON else 'OFF (text-only; set TTS_URL to speak)'}\n")

    if os.environ.get("CHAT_REPL") not in (None, "", "0"):
        # Interactive multi-turn loop -- keeps history so the persona/context carries across turns.
        messages = [{"role": "system", "content": SYSTEM}]
        print("Multi-turn chat (Ctrl-D or 'quit' to exit).")
        while True:
            try:
                user = input("you> ").strip()
            except EOFError:
                print()
                break
            if user in ("quit", "exit"):
                break
            if not user:
                continue
            messages.append({"role": "user", "content": user})
            reply = converse(messages)
            messages.append({"role": "assistant", "content": reply})
            print(f"bot> {reply}\n")
        return

    # One-shot: text from CHAT_TEXT or stdin.
    text = os.environ.get("CHAT_TEXT")
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    text = (text or "Tell me a tiny story about a wombat.").strip()
    print(f"you> {text}")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]
    reply = converse(messages)
    print(f"\nbot> {reply}")


if __name__ == "__main__":
    main()
