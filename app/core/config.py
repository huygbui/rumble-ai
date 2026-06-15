import os
from dataclasses import dataclass


def _flag(name: str) -> bool:
    return os.environ.get(name) not in (None, "", "0")


@dataclass(frozen=True, slots=True)
class Settings:
    llm_base: str
    llm_model: str
    max_tokens: int
    chat_system: str

    stt_base: str
    stt_model: str
    warm_budget: int

    tts_base: str
    tts_model: str
    tts_out_dir: str
    play: bool
    compare: bool
    omni_instructions: str
    omni_seed: int
    tts_voice: str
    say_max_len: int
    say_first_max: int
    say_min_len: int
    say_gap_ms: int
    say_fade_ms: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_base=os.environ.get("LLM_URL", "").rstrip("/"),
            llm_model=os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-4B"),
            max_tokens=int(os.environ.get("CHAT_MAX_TOKENS", "256")),
            chat_system=os.environ.get(
                "CHAT_SYSTEM",
                "You are a friendly Australian helper for kids. Speak simply and warmly, "
                "in one or two short sentences.",
            ),
            stt_base=os.environ.get("STT_URL", "").rstrip("/"),
            stt_model=os.environ.get("STT_MODEL", "Qwen/Qwen3-ASR-0.6B"),
            warm_budget=int(os.environ.get("WARM_BUDGET", "300")),
            tts_base=os.environ.get("TTS_URL", "").rstrip("/"),
            tts_model=os.environ.get("TTS_MODEL", "omnivoice").lower(),
            tts_out_dir=os.environ.get("TTS_OUT_DIR", "out"),
            play=_flag("PLAY"),
            compare=_flag("COMPARE"),
            omni_instructions=os.environ.get(
                "OMNI_INSTRUCTIONS",
                "female, child, high pitch, australian accent",
            ),
            omni_seed=int(os.environ.get("OMNI_SEED", "58842")),
            tts_voice=os.environ.get("TTS_VOICE", "bench"),
            say_max_len=int(os.environ.get("SAY_MAX_LEN", "140")),
            say_first_max=int(os.environ.get("SAY_FIRST_MAX", "60")),
            say_min_len=int(os.environ.get("SAY_MIN_LEN", "12")),
            say_gap_ms=int(os.environ.get("SAY_GAP_MS", "90")),
            say_fade_ms=int(os.environ.get("SAY_FADE_MS", "8")),
        )

    @property
    def llm_chat_url(self) -> str:
        return f"{self.llm_base}/v1/chat/completions"

    @property
    def tts_url(self) -> str:
        return f"{self.tts_base}/v1/audio/speech"

    @property
    def stt_on(self) -> bool:
        return bool(self.stt_base)

    @property
    def tts_on(self) -> bool:
        return bool(self.tts_base)


settings = Settings.from_env()
