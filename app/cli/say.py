import glob
import os
import sys

from app.core import speech


def main() -> None:
    if os.environ.get("STITCH_ONLY") not in (None, "", "0"):
        files = sorted(glob.glob(os.path.join(speech.OUT_DIR, "say_[0-9]*.wav")))
        if not files:
            raise SystemExit(f"no say_NN.wav clips in {speech.OUT_DIR}/ to stitch")
        clips = []
        for file in files:
            with open(file, "rb") as f:
                clips.append(f.read())
        out = speech.stitch(clips)
        sp = os.path.join(speech.OUT_DIR, "say.wav")
        with open(sp, "wb") as f:
            f.write(out)
        print(
            f"stitched {len(files)} clips -> {sp} ({speech.wav_dur(out):.2f}s, "
            f"gap={speech.GAP_MS}ms fade={speech.FADE_MS}ms)"
        )
        return

    text = os.environ.get("TTS_TEXT")
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    text = (text or "G'day! Want to hear a quick story? It begins on a windy hill by the sea.").strip()

    chunks = speech.split_clauses(text)
    print(f"endpoint={speech.URL}  model={speech.MODEL}  clauses={len(chunks)}\n")
    for i, c in enumerate(chunks):
        print(f"  clause[{i}] ({len(c)} chars): {c!r}")
    print()

    print("== chunked (pipelined) ==")
    speech.stream(chunks)

    if speech.COMPARE:
        print("\n== single-shot (whole text, one request) ==")
        tt, b = speech.synth(text)
        print(f"  first(&only) audio at {tt:.2f}s  |  {speech.wav_dur(b):.2f}s audio")
        print("  (single-shot TTFA == full synth time; chunked TTFA above is the win)")
