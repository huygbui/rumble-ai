import os
import sys

from app.core import dialogue, speech


def main() -> None:
    print(f"LLM={dialogue.LLM_CHAT_URL or '(unset)'}  model={dialogue.LLM_MODEL}")
    print(f"TTS={'on -> ' + speech.URL if dialogue.TTS_ON else 'OFF (text-only; set TTS_URL to speak)'}\n")

    if os.environ.get("CHAT_REPL") not in (None, "", "0"):
        messages = [{"role": "system", "content": dialogue.SYSTEM}]
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
            reply = dialogue.converse(messages)
            messages.append({"role": "assistant", "content": reply})
            print(f"bot> {reply}\n")
        return

    text = os.environ.get("CHAT_TEXT")
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    text = (text or "Tell me a tiny story about a wombat.").strip()
    print(f"you> {text}")
    messages = [
        {"role": "system", "content": dialogue.SYSTEM},
        {"role": "user", "content": text},
    ]
    reply = dialogue.converse(messages)
    print(f"\nbot> {reply}")


if __name__ == "__main__":
    main()
