# say.py
# Low-latency "say this line" for the TTS endpoints. Splits text into clauses and
# synthesizes them PIPELINED, so the first audio is ready after ~one short-clause synth
# instead of waiting for the whole utterance. This is the responsiveness shape for the
# back-and-forth chat loop: OmniVoice's server-side streaming barely helps (TTFA ~3.2s),
# but one short clause synthesizes in ~the per-request floor (~2.5s) and can start playing
# while the rest generate -- so felt latency tracks the FIRST clause, not the whole reply.
#
#   export TTS_URL="<flash url from `modal deploy tts/omnivoice.py`>"
#   export TTS_MODEL=omnivoice            # omnivoice | fish (fish needs a pre-registered TTS_VOICE)
#   echo "G'day! Want to hear a quick story? It begins on a windy hill." | python say.py
#   TTS_TEXT="..." python say.py          # or pass text via env
#   PLAY=1 ...                            # also play through speakers as chunks arrive (macOS afplay)
#   COMPARE=1 ...                         # also time single-shot (whole text, one request) for contrast
#
# Writes per-clause wavs to ./out (say_NN.wav) and prints time-to-first-audio (TTFA).
import array
import os
import queue
import re
import subprocess
import sys
import threading
import time
import io
import wave

import httpx

BASE = os.environ.get("TTS_URL", "").rstrip("/")  # required for synthesis; NOT for STITCH_ONLY
URL = f"{BASE}/v1/audio/speech"
MODEL = os.environ.get("TTS_MODEL", "omnivoice").lower()
OUT_DIR = os.environ.get("TTS_OUT_DIR", "out")
PLAY = os.environ.get("PLAY") not in (None, "", "0")
COMPARE = os.environ.get("COMPARE") not in (None, "", "0")
OMNI_INSTRUCTIONS = os.environ.get("OMNI_INSTRUCTIONS", "female, child, high pitch, australian accent")
OMNI_SEED = int(os.environ.get("OMNI_SEED", "58842"))  # one fixed seed -> consistent voice across clauses
VOICE = os.environ.get("TTS_VOICE", "bench")  # fish: a voice already registered on the server
MAX_LEN = int(os.environ.get("SAY_MAX_LEN", "140"))  # split clauses longer than this
FIRST_MAX = int(os.environ.get("SAY_FIRST_MAX", "60"))  # keep the FIRST clause short -> fast first audio
MIN_LEN = int(os.environ.get("SAY_MIN_LEN", "12"))      # fold tiny fragments into a neighbor
# --- Join smoothing (fixes the audible seam between separately-synthesized clauses) ------
GAP_MS = int(os.environ.get("SAY_GAP_MS", "90"))    # ONE controlled pause between clauses
FADE_MS = int(os.environ.get("SAY_FADE_MS", "8"))   # edge fade in/out -> no boundary click
SILENCE_THR = 400  # |s16| <= this counts as silence when trimming clip edges

CLIENT = httpx.Client(timeout=600, limits=httpx.Limits(max_keepalive_connections=8))  # keep-alive across clauses


def make_payload(text: str) -> dict:
    if MODEL == "omnivoice":
        return {"input": text, "instructions": OMNI_INSTRUCTIONS,
                "language": "English", "response_format": "wav", "seed": OMNI_SEED}
    if MODEL == "fish":
        return {"input": text, "voice": VOICE, "response_format": "wav", "seed": 58842}
    raise SystemExit(f"say.py supports TTS_MODEL=omnivoice|fish; got {MODEL!r}")


def synth(text: str):
    if not BASE:
        raise SystemExit("set TTS_URL to synthesize (only STITCH_ONLY works without it)")
    t0 = time.time()
    r = CLIENT.post(URL, json=make_payload(text))
    r.raise_for_status()
    return time.time() - t0, r.content


def wav_dur(b: bytes) -> float:
    try:
        w = wave.open(io.BytesIO(b))
        return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


# --- Seam smoothing: trim edge silence, fade edges, join with ONE controlled pause -------
# The audible seam between clauses is mostly (a) each clip's leading/trailing SILENCE
# doubling up at the join, (b) the player gap from playing separate files, and (c) edge
# clicks. Stitching into ONE stream with trimmed edges + a short fade + a single ~90ms pause
# removes (a)/(b)/(c). What it can't fix is cross-clause PROSODY (each clause is synthesized
# independently) -- but between SENTENCES a reset is natural, so this gets most of the way.

def _read_pcm(b: bytes):
    w = wave.open(io.BytesIO(b))
    a = array.array("h")
    a.frombytes(w.readframes(w.getnframes()))
    return w.getframerate(), a


def _write_pcm(sr: int, a: array.array) -> bytes:
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(a.tobytes())
    w.close()
    return buf.getvalue()


def _trim_and_fade(a: array.array, sr: int) -> array.array:
    n = len(a)
    if not n:
        return a
    start = next((i for i, x in enumerate(a) if abs(x) > SILENCE_THR), 0)
    end = n - next((i for i, x in enumerate(reversed(a)) if abs(x) > SILENCE_THR), 0)
    margin = int(sr * 0.015)  # keep 15ms either side so we don't clip the first/last phoneme
    a = array.array("h", a[max(0, start - margin):min(n, end + margin)])
    f = max(1, int(sr * FADE_MS / 1000))
    for i in range(min(f, len(a))):  # linear fade in/out -> no click at the join
        a[i] = int(a[i] * i / f)
        a[-1 - i] = int(a[-1 - i] * i / f)
    return a


def stitch(byte_list: list[bytes]) -> bytes:
    sr = 24000
    out = array.array("h")
    for j, b in enumerate(byte_list):
        sr, a = _read_pcm(b)
        if j and GAP_MS:
            out.extend(array.array("h", bytes(2 * int(sr * GAP_MS / 1000))))  # one pause
        out.extend(_trim_and_fade(a, sr))
    return _write_pcm(sr, out)


# Real sentence boundaries vs. periods inside numbers/abbreviations. Split on a terminator
# FOLLOWED BY whitespace (so "0.22.1" / "3.5" stay intact), then re-merge any split that landed
# right after a known abbreviation ("Dr. Smith", "p.m. sharp").
ABBREV = {"mr", "mrs", "ms", "dr", "st", "mt", "vs", "jr", "sr", "prof", "sgt", "gen",
          "rev", "hon", "co", "inc", "ltd", "etc", "e.g", "i.e", "a.m", "p.m"}


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for p in re.split(r"(?<=[.!?])\s+", text.strip()):
        p = p.strip()
        if not p:
            continue
        if out:
            toks = out[-1].split()
            tail = toks[-1].lower().rstrip(".") if toks else ""
            if tail in ABBREV:  # abbreviation, not a sentence end -> rejoin
                out[-1] = out[-1] + " " + p
                continue
        out.append(p)
    return out


def _merge_tiny(chunks: list[str], min_len: int = MIN_LEN) -> list[str]:
    # Fold sub-min_len fragments into a neighbor so we never pay the full per-request floor for
    # <0.5s of audio. A tiny opener merges FORWARD (its follower joins it); others fold into the
    # previous chunk. _shorten_first runs AFTER this, so a long merged opener is still re-shortened.
    out: list[str] = []
    for c in chunks:
        if out and len(out[-1]) < min_len:
            out[-1] = (out[-1] + " " + c).strip()
        else:
            out.append(c)
    if len(out) >= 2 and len(out[-1]) < min_len:  # trailing tiny -> fold back
        out[-2] = (out[-2] + " " + out.pop()).strip()
    return out


def _shorten_first(chunks: list[str], first_max: int = FIRST_MAX) -> list[str]:
    # A long opening sentence makes the first chunk ~= a full synth. Split it at the earliest
    # comma/semicolon/colon at least ~20 chars in -> a short opener -> fast first audio. If there
    # is no punctuation to split on, leave it intact (don't break mid-phrase).
    if not chunks or len(chunks[0]) <= first_max:
        return chunks
    head = chunks[0]
    m = next((mm for mm in re.finditer(r"[,;:]\s+", head) if mm.start() >= 20), None)
    if not m:
        return chunks
    first, rest = head[: m.start() + 1].strip(), head[m.end():].strip()
    return [first, rest] + chunks[1:]


def split_clauses(text: str, max_len: int = MAX_LEN) -> list[str]:
    # Sentence-split (decimal/abbrev safe); further split any over-long sentence on
    # comma/semicolon/colon; merge micro-fragments; then force a short FIRST chunk.
    chunks: list[str] = []
    for sent in _sentences(text):
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
    chunks = _merge_tiny(chunks)
    chunks = _shorten_first(chunks)
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
    collected: list = []
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
        collected.append((i, b))
        dur = wav_dur(b)
        total_audio += dur
        print(f"  [{i}] synth {tt:5.2f}s  audio {dur:5.2f}s  | {c[:55]!r}")
        if PLAY:
            subprocess.run(["afplay", path], check=False)
    wall = time.time() - t0
    print(f"  -- chunked total wall {wall:.2f}s for {total_audio:.2f}s audio "
          f"({len(chunks)} clauses)")
    # Smooth continuous render (trim/fade/one-pause joins) -> the seamless artifact to ship.
    if collected:
        stitched = stitch([b for _, b in sorted(collected)])
        sp = os.path.join(OUT_DIR, "say.wav")
        with open(sp, "wb") as f:
            f.write(stitched)
        print(f"  -- stitched -> {sp} ({wav_dur(stitched):.2f}s, gap={GAP_MS}ms fade={FADE_MS}ms)")


def main() -> None:
    # STITCH_ONLY: re-join existing out/say_NN.wav into a smooth out/say.wav, no synthesis
    # (free) -- for tuning GAP_MS/FADE_MS against already-generated clips.
    if os.environ.get("STITCH_ONLY") not in (None, "", "0"):
        import glob
        files = sorted(glob.glob(os.path.join(OUT_DIR, "say_[0-9]*.wav")))
        if not files:
            raise SystemExit(f"no say_NN.wav clips in {OUT_DIR}/ to stitch")
        out = stitch([open(f, "rb").read() for f in files])
        sp = os.path.join(OUT_DIR, "say.wav")
        with open(sp, "wb") as f:
            f.write(out)
        print(f"stitched {len(files)} clips -> {sp} ({wav_dur(out):.2f}s, "
              f"gap={GAP_MS}ms fade={FADE_MS}ms)")
        return

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
