import glob
import os
import queue
import subprocess
import sys
import threading
import time

from app.core import text_chunks, tts
from app.core.config import settings


def stitch_existing() -> None:
    files = sorted(glob.glob(os.path.join(settings.tts_out_dir, "say_[0-9]*.wav")))
    if not files:
        raise SystemExit(f"no say_NN.wav clips in {settings.tts_out_dir}/ to stitch")
    clips = []
    for file in files:
        with open(file, "rb") as f:
            clips.append(f.read())
    out = tts.stitch(clips)
    path = os.path.join(settings.tts_out_dir, "say.wav")
    with open(path, "wb") as f:
        f.write(out)
    print(
        f"stitched {len(files)} clips -> {path} ({tts.wav_duration(out):.2f}s, "
        f"gap={settings.say_gap_ms}ms fade={settings.say_fade_ms}ms)"
    )


def stream(chunks: list[str]) -> None:
    os.makedirs(settings.tts_out_dir, exist_ok=True)
    q: queue.Queue = queue.Queue()
    t0 = time.time()

    def producer():
        for i, text in enumerate(chunks):
            try:
                q.put((i, text, *tts.synthesize(text)))
            except Exception as e:
                q.put((i, text, e, None))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    first, total_audio, collected = None, 0.0, []
    while (item := q.get()) is not None:
        i, text, first_value, wav = item
        if wav is None:
            print(f"  [{i}] FAILED: {first_value}")
            continue
        synth_s = first_value
        if first is None:
            first = time.time() - t0
            print(f"  >> TTFA (first clause audio ready): {first:.2f}s")
        path = os.path.join(settings.tts_out_dir, f"say_{i:02d}.wav")
        with open(path, "wb") as f:
            f.write(wav)
        collected.append((i, wav))
        dur = tts.wav_duration(wav)
        total_audio += dur
        print(f"  [{i}] synth {synth_s:5.2f}s  audio {dur:5.2f}s  | {text[:55]!r}")
        if settings.play:
            subprocess.run(["afplay", path], check=False)

    wall = time.time() - t0
    print(f"  -- chunked total wall {wall:.2f}s for {total_audio:.2f}s audio ({len(chunks)} clauses)")
    if collected:
        stitched = tts.stitch([wav for _, wav in sorted(collected)])
        path = os.path.join(settings.tts_out_dir, "say.wav")
        with open(path, "wb") as f:
            f.write(stitched)
        print(f"  -- stitched -> {path} ({tts.wav_duration(stitched):.2f}s)")


def main() -> None:
    if os.environ.get("STITCH_ONLY") not in (None, "", "0"):
        stitch_existing()
        return

    text = os.environ.get("TTS_TEXT")
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    text = (text or "G'day! Want to hear a quick story? It begins on a windy hill by the sea.").strip()

    chunks = text_chunks.split_clauses(text)
    print(f"endpoint={settings.tts_speech_url}  model={settings.tts_model}  clauses={len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"  clause[{i}] ({len(chunk)} chars): {chunk!r}")
    print("\n== chunked (pipelined) ==")
    stream(chunks)

    if settings.compare:
        print("\n== single-shot (whole text, one request) ==")
        synth_s, wav = tts.synthesize(text)
        print(f"  first(&only) audio at {synth_s:.2f}s  |  {tts.wav_duration(wav):.2f}s audio")


if __name__ == "__main__":
    main()
