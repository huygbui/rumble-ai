# web.py
# Browser front-end for the chat.py voice loop, so you can FEEL the chunking win instead of
# reading TTFA numbers off a terminal. Same 3-stage overlap as chat.converse() -- LLM SSE
# stream -> ClauseStreamer -> say.synth -> ordered playback -- but each clause's WAV is pushed
# to the browser the instant it's synthesized, and the page plays them in order (Web Audio)
# while later clauses are still generating. So you hear clause 0 while clause 2 is mid-flight,
# and the page shows the head-start: how many seconds sooner audio began vs. the full reply.
#
# Pure stdlib (no FastAPI/uvicorn -- keeps the repo's modal+requests-only footprint). Reuses
# chat.py's ClauseStreamer/llm_stream and say.py's synth/wav_dur/stitch unchanged.
#
#   export LLM_URL="<flash url from `modal deploy llm/qwen3_5_4b.py`>"
#   export TTS_URL="<flash url from `modal deploy tts/omnivoice.py`>"     # omit -> text-only
#   export STT_URL="<flash url from `modal deploy stt/qwen3_asr.py`>"     # omit -> mic disabled
#   python web.py            # -> http://127.0.0.1:8000   (PORT=... to change)
#
# Transport: POST /api/chat with {messages:[...]} -> an SSE stream (text/event-stream, chunked)
# of `meta`/`ttft`/`clause`/`done`/`error` events. History lives in the browser (stateless server).
import base64
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

import chat  # reuse ClauseStreamer, llm_stream, LLM_* (importing runs only constant setup)
import say   # reuse synth(), wav_dur(), stitch(), OUT_DIR, BASE

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "web", "index.html")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
TTS_ON = bool(say.BASE)

# --- STT (front of the turn): OpenAI-compatible /v1/audio/transcriptions, like stt/qwen3_asr.py.
# Optional, same as TTS: unset -> the mic is disabled, you type instead. Set STT_URL to enable it.
STT_BASE = os.environ.get("STT_URL", "").rstrip("/")
STT_MODEL = os.environ.get("STT_MODEL", "Qwen/Qwen3-ASR-0.6B")
STT_ON = bool(STT_BASE)
STT_SESSION = requests.Session()  # keep-alive to the ASR host (its own, separate from LLM/TTS)
# Browser MediaRecorder hands us webm/ogg/mp4; vLLM[audio] decodes via soundfile + PyAV fallback.
STT_EXT = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a",
           "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav"}

# --- Readiness / warm-up: every stage scales to zero, so the first hit cold-starts. GPU snapshots
# make that fast (STT ~1-2s, TTS ~15s, LLM ~25s restore; minutes only on the first snapshot build) --
# but it's still jarring on a button tap, so /api/warm actively waits every stage up first.
WARM_BUDGET = int(os.environ.get("WARM_BUDGET", "300"))  # s, per-stage warm-up deadline
PROBE = requests.Session()  # /health probes across all three hosts


def _stages():
    # (name, base) for every CONFIGURED stage, in pipeline order (mic -> brain -> voice).
    s = []
    if STT_ON:
        s.append(("stt", STT_BASE))
    if chat.LLM_BASE:
        s.append(("llm", chat.LLM_BASE))
    if TTS_ON:
        s.append(("tts", say.BASE))
    return s


def _health_ok(base, timeout):
    # 200 from /health == ready. A scaled-to-zero endpoint returns 503/303 (or holds the request)
    # while it wakes, so one probe both REPORTS readiness AND nudges the container up; loop it
    # (warm_stage) to actually wait one up.
    try:
        return PROBE.get(f"{base}/health", timeout=timeout, allow_redirects=False).status_code == 200
    except Exception:
        return False


# --- One conversational turn, as a stream of (event, data) pairs -----------------------
# Mirrors chat.converse()'s voice/text loops, but yields SSE events instead of printing.
# llm_reader thread -> clause_q -> synth_worker thread -> the generator (this request thread)
# is the ordered consumer, exactly like chat.py's main loop.
def run_turn(messages):
    clause_q: queue.Queue = queue.Queue()
    t0 = time.time()
    state = {"ttft": None}

    def llm_reader():
        streamer = chat.ClauseStreamer()
        try:
            for delta in chat.llm_stream(messages):
                if state["ttft"] is None and delta.strip():
                    state["ttft"] = time.time() - t0
                    clause_q.put(("__TTFT__", state["ttft"]))
                for clause in streamer.feed(delta):
                    clause_q.put(clause)
            for clause in streamer.flush():
                clause_q.put(clause)
        except Exception as e:  # surface transport/LLM errors instead of hanging the stream
            clause_q.put(("__ERR__", f"{type(e).__name__}: {e}"))
        clause_q.put(None)

    threading.Thread(target=llm_reader, daemon=True).start()

    # --- text-only: no TTS_URL -> stream clauses as text, no audio (dev without the GPU) ---
    if not TTS_ON:
        i, parts = 0, []
        while True:
            item = clause_q.get()
            if item is None:
                break
            if isinstance(item, tuple):
                if item[0] == "__TTFT__":
                    yield "ttft", {"t": item[1]}
                else:
                    yield "error", {"message": item[1]}
                continue
            parts.append(item)
            yield "clause", {"i": i, "text": item, "t_ready": time.time() - t0,
                             "synth_s": None, "audio_s": None, "wav_b64": None}
            i += 1
        yield "done", {"wall": time.time() - t0, "total_audio": 0.0,
                       "n": len(parts), "full_reply": " ".join(parts)}
        return

    # --- voice mode: synth_worker pulls clauses, this generator emits ready audio in order ---
    audio_q: queue.Queue = queue.Queue()

    def synth_worker():
        i = 0
        while True:
            item = clause_q.get()
            if item is None:
                break
            if isinstance(item, tuple):  # pass TTFT/ERR markers straight through, in order
                audio_q.put(item)
                continue
            try:
                tt, b = say.synth(item)
                audio_q.put((i, item, b, tt))
                i += 1
            except Exception as e:
                audio_q.put(("__ERR__", f"{type(e).__name__}: {e}"))
        audio_q.put(None)

    threading.Thread(target=synth_worker, daemon=True).start()

    parts, collected, total_audio = [], [], 0.0
    while True:
        item = audio_q.get()
        if item is None:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] in ("__TTFT__", "__ERR__"):
            yield ("ttft", {"t": item[1]}) if item[0] == "__TTFT__" else ("error", {"message": item[1]})
            continue
        i, text, b, tt = item
        dur = say.wav_dur(b)
        total_audio += dur
        parts.append(text)
        collected.append((i, b))
        yield "clause", {"i": i, "text": text, "t_ready": time.time() - t0,
                         "synth_s": tt, "audio_s": dur,
                         "wav_b64": base64.b64encode(b).decode("ascii")}

    # Save the seamless stitched artifact too (parity with chat.py), best-effort.
    if collected:
        try:
            os.makedirs(say.OUT_DIR, exist_ok=True)
            stitched = say.stitch([b for _, b in sorted(collected)])
            with open(os.path.join(say.OUT_DIR, "web.wav"), "wb") as f:
                f.write(stitched)
        except Exception:
            pass
    yield "done", {"wall": time.time() - t0, "total_audio": total_audio,
                   "n": len(parts), "full_reply": " ".join(parts)}


# --- HTTP server -----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quieter console; turn-level metrics go to stdout below
        pass

    def _send(self, code, body: bytes, ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    # --- Chunked SSE (shared by /api/chat and /api/warm; chunked streams a body of unknown length)
    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, event, data):
        frame = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        self.wfile.write(f"{len(frame):X}\r\n".encode() + frame + b"\r\n")
        self.wfile.flush()

    def _sse_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                with open(INDEX_HTML, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, b"web/index.html missing")
        elif path == "/api/meta":
            meta = {"llm": chat.LLM_BASE or None, "model": chat.LLM_MODEL,
                    "tts": say.BASE or None, "tts_model": say.MODEL, "tts_on": TTS_ON,
                    "stt": STT_BASE or None, "stt_model": STT_MODEL, "stt_on": STT_ON}
            self._send(200, json.dumps(meta).encode(), "application/json")
        elif path == "/api/warm":  # actively wait every stage up, streaming each as it lands
            self.handle_warm()
        else:
            self._send(404, b"not found")

    # Warm every stage in parallel, emitting an SSE event as each becomes ready (or times out),
    # so the page can light up its status dots and only enable the buttons once all are up.
    def handle_warm(self):
        stages = _stages()
        self._sse_start()
        try:
            if not stages:
                self._sse("done", {"ready": False})
                self._sse_end()
                return
            q: queue.Queue = queue.Queue()
            t0 = time.time()

            def warm_stage(name, base):
                deadline = time.time() + WARM_BUDGET
                ok = False
                while time.time() < deadline:
                    if _health_ok(base, timeout=30):
                        ok = True
                        break
                    time.sleep(2)  # cold-start in progress -> re-probe (keeps nudging it up)
                q.put((name, ok, time.time() - t0))

            for n, b in stages:
                threading.Thread(target=warm_stage, args=(n, b), daemon=True).start()
            self._sse("warming", {"stages": [n for n, _ in stages]})
            all_ok = True
            for _ in stages:
                name, ok, t = q.get()
                all_ok = all_ok and ok
                print(f"  warm: {name} {'ready' if ok else 'FAILED'} in {t:.1f}s", flush=True)
                self._sse("stage", {"name": name, "status": "ready" if ok else "failed", "t": t})
            self._sse("done", {"ready": all_ok})
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError):
            pass  # page closed mid-warm; the warm threads finish on their own

    # Mic -> text. Browser POSTs the recorded audio as the raw body (Content-Type = its blob
    # type); we forward it as the OpenAI multipart `file` to the ASR endpoint and return {text}.
    # Always 200 with {text} or {error} so the page can handle both uniformly.
    def handle_stt(self):
        if not STT_ON:
            self._send(200, json.dumps({"error": "STT_URL is not set"}).encode(), "application/json")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            audio = self.rfile.read(n)
            ctype = (self.headers.get("Content-Type") or "audio/webm").split(";")[0].strip()
            text = self._asr(audio, ctype, STT_EXT.get(ctype, "webm"))
            print(f"  stt: {len(audio)} bytes -> {text[:60]!r}", flush=True)
            self._send(200, json.dumps({"text": text}).encode(), "application/json")
        except Exception as e:
            self._send(200, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    @staticmethod
    def _asr(audio, ctype, ext, attempts=4):
        # Cold start: while the scaled-to-zero ASR container wakes (snapshot restore ~1-2s, but the
        # first POST can race the wake), Modal proxies a 303 back instead of holding the request.
        # requests would downgrade that 303 to a body-less GET, so we set allow_redirects=False,
        # SEE the 303, and re-POST the same multipart until the container serves a 200.
        # (Verified 2026-06-15: cold 303 -> warm 200 with transcript.)
        last = ""
        for _ in range(attempts):
            r = STT_SESSION.post(
                f"{STT_BASE}/v1/audio/transcriptions",
                files={"file": (f"speech.{ext}", audio, ctype)},
                data={"model": STT_MODEL},  # language auto-detected; add "language":"en" to force
                timeout=600, allow_redirects=False,
            )
            if r.status_code == 200:
                return (r.json().get("text") or "").strip()
            last = f"{r.status_code} {r.text[:160]}"
            if r.status_code in (302, 303, 307, 308, 502, 503):  # cold-start handoff -> re-POST
                time.sleep(1.5)
                continue
            r.raise_for_status()
        raise RuntimeError(f"ASR not ready after {attempts} tries: {last}")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/stt":
            self.handle_stt()
            return
        if path != "/api/chat":
            self._send(404, b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages") or []
        except Exception as e:
            self._send(400, f"bad request: {e}".encode())
            return

        # Stream the turn as chunked SSE.
        self._sse_start()
        try:
            self._sse("meta", {"tts_on": TTS_ON})
            if not chat.LLM_BASE:
                self._sse("error", {"message": "LLM_URL is not set -- export it before starting web.py"})
            else:
                last = None
                for event, data in run_turn(messages):
                    self._sse(event, data)
                    if event == "done":
                        last = data
                if last:
                    print(f"  turn: {last['n']} clauses, {last['wall']:.2f}s wall, "
                          f"{last['total_audio']:.2f}s audio", flush=True)
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-stream


def main():
    print(f"STT = {(STT_BASE + '  model=' + STT_MODEL) if STT_ON else 'OFF (mic disabled; export STT_URL to speak in)'}")
    print(f"LLM = {chat.LLM_BASE or '(unset -- export LLM_URL)'}  model={chat.LLM_MODEL}")
    print(f"TTS = {(say.BASE + '  model=' + say.MODEL) if TTS_ON else 'OFF (text-only; export TTS_URL to speak)'}")
    print(f"\n  open  ->  http://{HOST}:{PORT}\n")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
