import asyncio
import base64
import os
import subprocess
import sys

import httpx

from app.core import pipeline, tts
from app.core.config import settings


async def converse(client: httpx.AsyncClient, messages: list[dict]) -> str:
    if not settings.llm_url:
        raise SystemExit("set LLM_URL to the Qwen3.5-4B endpoint (see llm/qwen3_5_4b.py)")

    if settings.tts_on:
        os.makedirs(settings.tts_out_dir, exist_ok=True)
    collected, parts, first_audio = [], [], False
    final = {}
    async for event in pipeline.run_turn(client, messages):
        data = event.data
        if event.event == "ttft":
            print(f"  >> LLM TTFT {data['t']:.2f}s")
        elif event.event == "clause":
            text = data["text"]
            parts.append(text)
            if "wav_b64" in data:
                wav = base64.b64decode(data["wav_b64"])
                path = os.path.join(settings.tts_out_dir, f"chat_{data['i']:02d}.wav")
                with open(path, "wb") as f:
                    f.write(wav)
                collected.append((data["i"], wav))
                if not first_audio:
                    first_audio = True
                    print(f"  >> first audio {data['t_ready']:.2f}s")
                print(f"  [{data['i']}] synth {data['synth_s']:5.2f}s  audio {data['audio_s']:5.2f}s  | {text[:55]!r}")
                if settings.play:
                    subprocess.run(["afplay", path], check=False)
            else:
                print(f"  [{data['i']}] {text}")
        elif event.event == "error":
            print(f"  [error] {data['message']}")
        elif event.event == "done":
            final = data

    if collected:
        stitched = tts.stitch([wav for _, wav in sorted(collected)])
        path = os.path.join(settings.tts_out_dir, "chat.wav")
        with open(path, "wb") as f:
            f.write(stitched)
        print(f"  -- stitched -> {path} ({tts.wav_duration(stitched):.2f}s)")
    if final:
        print(f"  -- turn wall {final['wall']:.2f}s for {final['total_audio']:.2f}s audio ({final['n']} clauses)")
    return final.get("full_reply") or " ".join(parts)


async def run() -> None:
    print(f"LLM={settings.llm_chat_url or '(unset)'}  model={settings.llm_model}")
    tts = f"on -> {settings.tts_speech_url}" if settings.tts_on else "OFF (text-only; set TTS_URL to speak)"
    print(f"TTS={tts}\n")

    async with httpx.AsyncClient(timeout=600, follow_redirects=False) as client:
        if os.environ.get("CHAT_REPL") not in (None, "", "0"):
            messages = [{"role": "system", "content": settings.chat_system}]
            print("Multi-turn chat (Ctrl-D or 'quit' to exit).")
            while True:
                try:
                    user = input("you> ").strip()
                except EOFError:
                    print()
                    break
                if user in ("quit", "exit"):
                    break
                if not user:
                    continue
                messages.append({"role": "user", "content": user})
                reply = await converse(client, messages)
                messages.append({"role": "assistant", "content": reply})
                print(f"bot> {reply}\n")
            return

        text = os.environ.get("CHAT_TEXT")
        if not text and not sys.stdin.isatty():
            text = sys.stdin.read()
        text = (text or "Tell me a tiny story about a wombat.").strip()
        print(f"you> {text}")
        messages = [
            {"role": "system", "content": settings.chat_system},
            {"role": "user", "content": text},
        ]
        print(f"\nbot> {await converse(client, messages)}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
