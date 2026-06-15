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
#   export TTS_URL="https://<workspace>--omnivoice-tts-serve.modal.run"   # omit -> text-only
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

import chat  # reuse ClauseStreamer, llm_stream, LLM_* (importing runs only constant setup)
import say   # reuse synth(), wav_dur(), stitch(), OUT_DIR, BASE

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "web", "index.html")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
TTS_ON = bool(say.BASE)


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
                    "tts": say.BASE or None, "tts_model": say.MODEL, "tts_on": TTS_ON}
            self._send(200, json.dumps(meta).encode(), "application/json")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/chat":
            self._send(404, b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages") or []
        except Exception as e:
            self._send(400, f"bad request: {e}".encode())
            return

        # Stream the turn as chunked SSE. (Chunked so HTTP/1.1 can stream a body of unknown length.)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event, data):
            frame = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
            self.wfile.write(f"{len(frame):X}\r\n".encode() + frame + b"\r\n")
            self.wfile.flush()

        try:
            emit("meta", {"tts_on": TTS_ON})
            if not chat.LLM_BASE:
                emit("error", {"message": "LLM_URL is not set -- export it before starting web.py"})
            else:
                last = None
                for event, data in run_turn(messages):
                    emit(event, data)
                    if event == "done":
                        last = data
                if last:
                    print(f"  turn: {last['n']} clauses, {last['wall']:.2f}s wall, "
                          f"{last['total_audio']:.2f}s audio", flush=True)
            self.wfile.write(b"0\r\n\r\n")  # terminating chunk
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-stream


def main():
    print(f"LLM = {chat.LLM_BASE or '(unset -- export LLM_URL)'}  model={chat.LLM_MODEL}")
    print(f"TTS = {(say.BASE + '  model=' + say.MODEL) if TTS_ON else 'OFF (text-only; export TTS_URL to speak)'}")
    print(f"\n  open  ->  http://{HOST}:{PORT}\n")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
