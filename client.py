# client.py
# Call the deployed Fish Speech S2 Pro endpoint twice:
#   (a) plain text-to-speech            -> tts.wav
#   (b) voice cloning (ref_audio+ref_text) -> cloned.wav
#
# The cloning fields (ref_audio/ref_text/seed) are vLLM-Omni extensions not in the
# typed OpenAI SDK, so we POST raw JSON to /v1/audio/speech with `requests`.
#
#   export TTS_URL="https://<workspace>--fish-s2-pro-tts-serve.modal.run"
#   python client.py
#
#   # voice cloning needs a reference clip + its exact transcript.
#   # Against a REMOTE Modal endpoint, REF_AUDIO must be either:
#   #   - a publicly reachable https:// URL, OR
#   #   - a data:audio/wav;base64,... data URI, OR
#   #   - a LOCAL file path: this client auto-encodes it into a data: URI before POST.
#   # A bare local path forwarded verbatim only works if that path exists INSIDE the
#   # serving container, which it does not for a remote deployment -- hence the
#   # client-side encoding below.
#   export REF_AUDIO="./reference.wav"   # local path, https:// URL, or data: URI
#   export REF_TEXT="The exact transcript of the reference audio."

import base64
import mimetypes
import os

import requests

BASE_URL = os.environ["TTS_URL"].rstrip("/")  # printed by `modal deploy`
SPEECH_URL = f"{BASE_URL}/v1/audio/speech"


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


def tts() -> None:
    # Plain text-to-speech. seed -> deterministic voice (PR #2624); if the deployed
    # build predates that PR, Pydantic silently drops the field. voice="default".
    synthesize(
        {
            "input": "Hello, this is Fish Speech S2 Pro running on Modal.",
            "voice": "default",
            "response_format": "wav",
            "seed": 58842,
        },
        "tts.wav",
    )


def clone() -> None:
    # Voice cloning (Base mode): BOTH ref_audio AND ref_text are required for s2-pro.
    # We omit task_type: every research cloning example passes only ref_audio +
    # ref_text and lets Base mode be inferred.
    ref_audio = os.environ.get("REF_AUDIO")
    ref_text = os.environ.get("REF_TEXT")
    if not ref_audio or not ref_text:
        print("skip clone: set REF_AUDIO + REF_TEXT to run voice cloning")
        return
    synthesize(
        {
            "input": "This sentence is spoken in the cloned reference voice.",
            "voice": "default",
            "response_format": "wav",
            "ref_audio": resolve_ref_audio(ref_audio),  # https URL or data: URI
            "ref_text": ref_text,  # exact transcript of the reference clip
        },
        "cloned.wav",
    )


if __name__ == "__main__":
    tts()
    clone()
