# client.py
# Generalized client for the OpenAI-compatible /v1/audio/speech endpoints in this repo.
# Picks the right per-model request shape via TTS_MODEL, so one tool can A/B all three:
#
#   export TTS_URL="https://<workspace>--<app>-serve.modal.run"
#   export TTS_MODEL=omnivoice   # fish | omnivoice | qwen-customvoice | qwen-base | qwen-voicedesign
#   python client.py
#
# Optional overrides:
#   export TTS_TEXT="G'day! Want to hear a quick story?"   # the line to speak
#   export TTS_VOICE=vivian                                # preset speaker (qwen-customvoice)
#   export REF_AUDIO=./reference.wav   # local path, https:// URL, or data: URI (cloning)
#   export REF_TEXT="exact transcript of the reference clip"
#   export TTS_OUT_DIR=out             # where generated .wav files are written (default: out/)
#
# Per-model voice handling differs:
#   fish             -> NO built-in voice: registers a reusable voice from REF_AUDIO, then
#                       synthesizes by name; also does inline cloning. REQUIRES REF_AUDIO+REF_TEXT.
#   omnivoice        -> voice DESIGN needs no reference (Australian-accent kid voice from
#                       `instructions`); cloning runs too if REF_AUDIO+REF_TEXT are set.
#   qwen-customvoice -> preset speaker (TTS_VOICE) + language + optional style instructions.
#   qwen-base        -> zero-shot cloning. REQUIRES REF_AUDIO+REF_TEXT.
#   qwen-voicedesign -> design a voice from a natural-language `instructions` prompt.
#
# Against a REMOTE Modal endpoint, an inline ref_audio must be reachable by the SERVER: a
# public https:// URL, a data:audio/...;base64,... URI, or a local file (this client
# auto-encodes a local path into a data: URI before POST). Fish voice REGISTRATION uploads
# the bytes directly (multipart), so it needs a LOCAL file.

import io
import base64
import mimetypes
import os
import wave

import httpx

BASE_URL = os.environ["TTS_URL"].rstrip("/")  # printed by `modal deploy`
SPEECH_URL = f"{BASE_URL}/v1/audio/speech"
VOICES_URL = f"{BASE_URL}/v1/audio/voices"  # fish-only (vllm-omni voice registry)

MODEL = os.environ.get("TTS_MODEL", "fish").lower()
TEXT = os.environ.get("TTS_TEXT", "G'day! Do you want to hear a quick story?")
VOICE = os.environ.get("TTS_VOICE", "vivian")  # preset speaker for qwen-customvoice
REF_AUDIO = os.environ.get("REF_AUDIO")
REF_TEXT = os.environ.get("REF_TEXT")
OUT_DIR = os.environ.get("TTS_OUT_DIR", "out")  # generated audio lands here (gitignored)


def _wav_dur(b: bytes) -> str:
    # Best-effort duration readout so a "success" really means we got decodable audio.
    try:
        w = wave.open(io.BytesIO(b))
        return f"{w.getnframes() / w.getframerate():.2f}s @ {w.getframerate()}Hz"
    except Exception:
        return "non-WAV body"


def synthesize(label: str, payload: dict, out_name: str) -> None:
    resp = httpx.post(SPEECH_URL, json=payload, timeout=600)
    if resp.is_error:
        # Surface the server's error body (these endpoints return JSON errors, e.g. a bad
        # preset voice or a missing ref) instead of an opaque status code.
        raise SystemExit(f"[{label}] {resp.status_code}: {resp.text[:500]}")
    os.makedirs(OUT_DIR, exist_ok=True)  # created on demand; OUT_DIR defaults to ./out
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"[{label}] wrote {out_path}  ({len(resp.content)} bytes, {_wav_dur(resp.content)})")


def resolve_ref_audio(ref_audio: str) -> str:
    # A caller's local file does not exist inside the Modal container, so encode a local
    # path into a data: URI here, client-side. URLs and data: URIs pass through unchanged.
    if ref_audio.startswith(("http://", "https://", "data:")):
        return ref_audio
    if os.path.isfile(ref_audio):
        mime = mimetypes.guess_type(ref_audio)[0] or "audio/wav"
        with open(ref_audio, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return ref_audio  # not a URL/data/local file -> forward as-is, let the server decide


def register_voice(name: str, audio_path: str, ref_text: str) -> None:
    # Fish-only: upload a reusable voice via multipart POST /v1/audio/voices, then
    # synthesize with voice=<name>. Re-registering the same name overwrites it.
    mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    with open(audio_path, "rb") as f:
        resp = httpx.post(
            VOICES_URL,
            data={
                "name": name,
                "consent": "I consent to using this reference voice for synthesis.",
                "ref_text": ref_text,
            },
            files={"audio_sample": (os.path.basename(audio_path), f, mime)},
            timeout=120,
        )
    resp.raise_for_status()
    print(f"registered voice {name!r}: success={resp.json().get('success')}")


def _have_ref() -> bool:
    return bool(REF_AUDIO and REF_TEXT)


def build_jobs() -> list[tuple[str, dict, str]]:
    # Returns a list of (label, payload, out_name). May raise SystemExit for models that
    # cannot synthesize anything without a reference clip (fish, qwen-base).
    if MODEL == "omnivoice":
        # Voice design needs NO reference -> a true end-to-end test from scratch. Accent /
        # age / pitch go in `instructions`; non-verbals like [laughter] go inline in input.
        jobs = [(
            "omnivoice:design(AU child)",
            {
                "input": "G'day! Want to hear a quick story? [laughter]",
                "instructions": "child, australian accent, high pitch",
                "language": "English",
                "response_format": "wav",
            },
            "omnivoice_design.wav",
        )]
        if _have_ref():
            jobs.append((
                "omnivoice:clone",
                {
                    "input": TEXT,
                    "ref_audio": resolve_ref_audio(REF_AUDIO),
                    "ref_text": REF_TEXT,
                    "language": "English",
                    "response_format": "wav",
                },
                "omnivoice_clone.wav",
            ))
        return jobs

    if MODEL == "qwen-customvoice":
        return [(
            f"qwen-customvoice:{VOICE}",
            {
                "input": TEXT,
                "voice": VOICE,  # preset speaker; override via TTS_VOICE
                "language": "English",
                "instructions": "speak in a warm, friendly tone",
                "response_format": "wav",
            },
            "qwen_customvoice.wav",
        )]

    if MODEL == "qwen-voicedesign":
        return [(
            "qwen-voicedesign",
            {
                "task_type": "VoiceDesign",
                "instructions": "A cheerful young Australian woman, gentle and clear",
                "input": TEXT,
                "language": "English",
                "response_format": "wav",
            },
            "qwen_voicedesign.wav",
        )]

    if MODEL == "qwen-base":
        if not _have_ref():
            raise SystemExit("qwen-base needs cloning input: set REF_AUDIO + REF_TEXT")
        return [(
            "qwen-base:clone",
            {
                "task_type": "Base",
                "input": TEXT,
                "ref_audio": resolve_ref_audio(REF_AUDIO),
                "ref_text": REF_TEXT,
                "language": "English",
                "response_format": "wav",
            },
            "qwen_base_clone.wav",
        )]

    if MODEL == "fish":
        if not _have_ref():
            raise SystemExit("fish has no built-in voice: set REF_AUDIO + REF_TEXT")
        jobs = []
        # Path (a): register a reusable voice (needs a LOCAL file), synthesize by name.
        if os.path.isfile(REF_AUDIO):
            register_voice("myvoice", REF_AUDIO, REF_TEXT)
            jobs.append((
                "fish:registered",
                {"input": TEXT, "voice": "myvoice", "response_format": "wav", "seed": 58842},
                "tts.wav",
            ))
        # Path (b): inline cloning (works with local path / URL / data: URI).
        jobs.append((
            "fish:clone",
            {
                "input": "This sentence is spoken in the cloned reference voice.",
                "ref_audio": resolve_ref_audio(REF_AUDIO),
                "ref_text": REF_TEXT,
                "response_format": "wav",
                "seed": 58842,
            },
            "cloned.wav",
        ))
        return jobs

    raise SystemExit(
        f"unknown TTS_MODEL={MODEL!r}; expected one of: "
        "fish | omnivoice | qwen-customvoice | qwen-base | qwen-voicedesign"
    )


if __name__ == "__main__":
    print(f"endpoint={SPEECH_URL}  model={MODEL}\n")
    for label, payload, out_path in build_jobs():
        synthesize(label, payload, out_path)
