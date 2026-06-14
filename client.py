# client.py
# Call the deployed Fish Speech S2 Pro endpoint.
#
# S2 Pro is a ZERO-SHOT voice model: it has NO built-in speakers, so every speech
# request must supply a voice in ONE of two ways --
#   (a) Registered voice  -> upload a reference clip once via POST /v1/audio/voices,
#                            then synthesize with voice="<name>" (no per-request ref).
#   (b) Inline cloning     -> pass ref_audio + ref_text on each request (Base mode).
# The pre-0.22 `voice:"default"` shortcut was REMOVED upstream and now returns 400
# ("Fish Speech has no built-in speakers").
#
#   export TTS_URL="https://<workspace>--fish-s2-pro-tts-serve.modal.run"
#   # A reference clip (10-30s) + its EXACT transcript are required for both paths:
#   export REF_AUDIO="./reference.wav"   # local path, https:// URL, or data: URI
#   export REF_TEXT="The exact transcript of the reference audio."
#   python client.py
#   # -> tts.wav     (path a: registered voice, if REF_AUDIO is a local file)
#   # -> cloned.wav  (path b: inline ref_audio + ref_text)
#
# Against a REMOTE Modal endpoint, inline ref_audio must be reachable by the SERVER:
# a public https:// URL, a data:audio/...;base64,... URI, or a local file (this
# client auto-encodes a local path into a data: URI before POST). Voice REGISTRATION
# uploads the bytes directly (multipart), so it needs a LOCAL file.

import base64
import mimetypes
import os

import requests

BASE_URL = os.environ["TTS_URL"].rstrip("/")  # printed by `modal deploy`
SPEECH_URL = f"{BASE_URL}/v1/audio/speech"
VOICES_URL = f"{BASE_URL}/v1/audio/voices"


def synthesize(payload: dict, out_path: str) -> None:
    # response_format=wav -> s2-pro returns 44.1 kHz mono WAV (binary body).
    resp = requests.post(SPEECH_URL, json=payload, timeout=600)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"wrote {out_path} ({len(resp.content)} bytes)")


def resolve_ref_audio(ref_audio: str) -> str:
    # vllm-omni's "local path -> auto base64" resolves the path SERVER-SIDE (inside
    # the Modal container), where a caller's local file does not exist. So if
    # REF_AUDIO points at a local file, encode it into a data: URI here, client-side.
    # URLs and existing data: URIs are passed through unchanged.
    if ref_audio.startswith(("http://", "https://", "data:")):
        return ref_audio
    if os.path.isfile(ref_audio):
        mime = mimetypes.guess_type(ref_audio)[0] or "audio/wav"
        with open(ref_audio, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    # Not a URL, not a data: URI, not a local file -> forward as-is and let the
    # server decide (will likely fail if it is a non-existent path).
    return ref_audio


def register_voice(name: str, audio_path: str, ref_text: str) -> None:
    # Upload a reusable voice via multipart POST /v1/audio/voices. Required fields:
    # name, consent (free-text), plus the audio_sample bytes + its ref_text. After
    # this, synthesize with voice=<name> and no per-request reference. Re-registering
    # the same name overwrites it.
    mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    with open(audio_path, "rb") as f:
        resp = requests.post(
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


def tts() -> None:
    # Path (a): "plain" TTS via a pre-registered voice. Needs a LOCAL ref file to
    # upload; for a remote-only ref (URL/data URI) use clone() instead. seed ->
    # deterministic voice.
    ref_audio = os.environ.get("REF_AUDIO")
    ref_text = os.environ.get("REF_TEXT")
    if not ref_audio or not ref_text:
        print("skip tts: set REF_AUDIO + REF_TEXT (S2 Pro has no built-in voice)")
        return
    if not os.path.isfile(ref_audio):
        print("skip tts: REF_AUDIO is not a local file; registration needs local bytes (clone() still runs)")
        return
    register_voice("myvoice", ref_audio, ref_text)
    synthesize(
        {
            "input": "Hello, this is Fish Speech S2 Pro running on Modal.",
            "voice": "myvoice",
            "response_format": "wav",
            "seed": 58842,
        },
        "tts.wav",
    )


def clone() -> None:
    # Path (b): inline voice cloning (Base mode). BOTH ref_audio AND ref_text are
    # required. Works with a local path, https:// URL, or data: URI.
    ref_audio = os.environ.get("REF_AUDIO")
    ref_text = os.environ.get("REF_TEXT")
    if not ref_audio or not ref_text:
        print("skip clone: set REF_AUDIO + REF_TEXT to run voice cloning")
        return
    synthesize(
        {
            "input": "This sentence is spoken in the cloned reference voice.",
            "response_format": "wav",
            "ref_audio": resolve_ref_audio(ref_audio),  # local path / https URL / data: URI
            "ref_text": ref_text,  # exact transcript of the reference clip
            "seed": 58842,
        },
        "cloned.wav",
    )


if __name__ == "__main__":
    tts()
    clone()
