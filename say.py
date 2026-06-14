# say.py
# Low-latency "say this line" for the TTS endpoints. Splits text into clauses and
# synthesizes them PIPELINED, so the first audio is ready after ~one short-clause synth
# instead of waiting for the whole utterance. This is the responsiveness shape for the
# back-and-forth chat loop: OmniVoice's server-side streaming barely helps (TTFA ~3.2s),
# but one short clause synthesizes in ~the per-request floor (~2.5s) and can start playing
# while the rest generate -- so felt latency tracks the FIRST clause, not the whole reply.
#
#   export TTS_URL="https://<workspace>--omnivoice-tts-serve.modal.run"
#   export TTS_MODEL=omnivoice            # omnivoice | fish (fish needs a pre-registered TTS_VOICE)
#   echo "G'day! Want to hear a quick story? It begins on a windy hill." | python say.py
#   TTS_TEXT="..." python say.py          # or pass text via env
#   PLAY=1 ...                            # also play through speakers as chunks arrive (macOS afplay)
#   COMPARE=1 ...                         # also time single-shot (whole text, one request) for contrast
#
# Writes per-clause wavs to ./out (say_NN.wav) and prints time-to-first-audio (TTFA).
import os
import queue
import re
import subprocess
import sys
import threading
import time
import io
import wave

import requests

BASE = os.environ["TTS_URL"].rstrip("/")
URL = f"{BASE}/v1/audio/speech"
MODEL = os.environ.get("TTS_MODEL", "omnivoice").lower()
OUT_DIR = os.environ.get("TTS_OUT_DIR", "out")
PLAY = os.environ.get("PLAY") not in (None, "", "0")
COMPARE = os.environ.get("COMPARE") not in (None, "", "0")
OMNI_INSTRUCTIONS = os.environ.get("OMNI_INSTRUCTIONS", "child, australian accent, high pitch")
VOICE = os.environ.get("TTS_VOICE", "bench")  # fish: a voice already registered on the server
MAX_LEN = int(os.environ.get("SAY_MAX_LEN", "140"))  # split clauses longer than this


def make_payload(text: str) -> dict:
    if MODEL == "omnivoice":
        return {"input": text, "instructions": OMNI_INSTRUCTIONS,
                "language": "English", "response_format": "wav"}
    if MODEL == "fish":
        return {"input": text, "voice": VOICE, "response_format": "wav", "seed": 58842}
    raise SystemExit(f"say.py supports TTS_MODEL=omnivoice|fish; got {MODEL!r}")


def synth(text: str):
    t0 = time.time()
    r = requests.post(URL, json=make_payload(text), timeout=600)
    r.raise_for_status()
    return time.time() - t0, r.content


def wav_dur(b: bytes) -> float:
    try:
        w = wave.open(io.BytesIO(b))
        return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def split_clauses(text: str, max_len: int = MAX_LEN) -> list[str]:
    # Sentence-split (keep the terminator); further split any over-long sentence on
    # comma/semicolon/colon so the FIRST chunk is short -> fast first audio.
    chunks: list[str] = []
    for sent in re.findall(r"[^.!?]+[.!?]?", text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_len:
            chunks.append(sent)
            continue
        buf = ""
        for part in re.split(r"(?<=[,;:])\s+", sent):
            if len(buf) + len(part) + 1 <= max_len:
                buf = (buf + " " + part).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = part
        if buf:
            chunks.append(buf)
    return chunks


def stream(chunks: list[str]) -> None:
    # Producer thread synthesizes clauses in order onto a queue; the main thread consumes
    # (and optionally plays) in order -> clause N+1 synthesizes while clause N plays.
    os.makedirs(OUT_DIR, exist_ok=True)
    q: queue.Queue = queue.Queue()
    t0 = time.time()

    def producer():
        for i, c in enumerate(chunks):
            try:
                tt, b = synth(c)
                q.put((i, c, b, tt))
            except Exception as e:  # one failed clause shouldn't wedge the consumer
                q.put((i, c, None, e))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    first = None
    total_audio = 0.0
    while True:
        item = q.get()
        if item is None:
            break
        i, c, b, tt = item
        if b is None:
            print(f"  [{i}] FAILED: {tt}")
            continue
        if first is None:
            first = time.time() - t0
            print(f"  >> TTFA (first clause audio ready): {first:.2f}s")
        path = os.path.join(OUT_DIR, f"say_{i:02d}.wav")
        with open(path, "wb") as f:
            f.write(b)
        dur = wav_dur(b)
        total_audio += dur
        print(f"  [{i}] synth {tt:5.2f}s  audio {dur:5.2f}s  | {c[:55]!r}")
        if PLAY:
            subprocess.run(["afplay", path], check=False)
    wall = time.time() - t0
    print(f"  -- chunked total wall {wall:.2f}s for {total_audio:.2f}s audio "
          f"({len(chunks)} clauses)")


def main() -> None:
    text = os.environ.get("TTS_TEXT")
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    text = (text or "G'day! Want to hear a quick story? It begins on a windy hill by the sea.").strip()

    chunks = split_clauses(text)
    print(f"endpoint={URL}  model={MODEL}  clauses={len(chunks)}\n")
    for i, c in enumerate(chunks):
        print(f"  clause[{i}] ({len(c)} chars): {c!r}")
    print()

    print("== chunked (pipelined) ==")
    stream(chunks)

    if COMPARE:
        print("\n== single-shot (whole text, one request) ==")
        tt, b = synth(text)
        print(f"  first(&only) audio at {tt:.2f}s  |  {wav_dur(b):.2f}s audio")
        print("  (single-shot TTFA == full synth time; chunked TTFA above is the win)")


if __name__ == "__main__":
    main()
