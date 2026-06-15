import base64
import io
import mimetypes
import os
import wave

import httpx

BASE_URL = os.environ["TTS_URL"].rstrip("/")
SPEECH_URL = f"{BASE_URL}/v1/audio/speech"

MODEL = os.environ.get("TTS_MODEL", "omnivoice").lower()
TEXT = os.environ.get("TTS_TEXT", "G'day! Do you want to hear a quick story?")
VOICE = os.environ.get("TTS_VOICE", "vivian")
REF_AUDIO = os.environ.get("REF_AUDIO")
REF_TEXT = os.environ.get("REF_TEXT")
OUT_DIR = os.environ.get("TTS_OUT_DIR", "out")


def _wav_dur(b: bytes) -> str:
    try:
        w = wave.open(io.BytesIO(b))
        return f"{w.getnframes() / w.getframerate():.2f}s @ {w.getframerate()}Hz"
    except Exception:
        return "non-WAV body"


def synthesize(label: str, payload: dict, out_name: str) -> None:
    resp = httpx.post(SPEECH_URL, json=payload, timeout=600)
    if resp.is_error:
        raise SystemExit(f"[{label}] {resp.status_code}: {resp.text[:500]}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"[{label}] wrote {out_path}  ({len(resp.content)} bytes, {_wav_dur(resp.content)})")


def resolve_ref_audio(ref_audio: str) -> str:
    if ref_audio.startswith(("http://", "https://", "data:")):
        return ref_audio
    if os.path.isfile(ref_audio):
        mime = mimetypes.guess_type(ref_audio)[0] or "audio/wav"
        with open(ref_audio, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return ref_audio


def _have_ref() -> bool:
    return bool(REF_AUDIO and REF_TEXT)


def build_jobs() -> list[tuple[str, dict, str]]:
    if MODEL == "omnivoice":
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

    raise SystemExit(
        f"unknown TTS_MODEL={MODEL!r}; expected one of: "
        "omnivoice | qwen-customvoice | qwen-base | qwen-voicedesign"
    )


if __name__ == "__main__":
    print(f"endpoint={SPEECH_URL}  model={MODEL}\n")
    for label, payload, out_path in build_jobs():
        synthesize(label, payload, out_path)
