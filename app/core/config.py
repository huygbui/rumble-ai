from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CHAT_SYSTEM = (
    "You are a friendly Australian helper for kids. Speak simply and warmly, "
    "in one or two short sentences."
)


def strip_url(value: str | None) -> str:
    return (value or "").rstrip("/")


def blank_flag_is_false(value: object) -> object:
    return False if value in (None, "") else value


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    llm_url: str = ""
    llm_model: str = "Qwen/Qwen3.5-4B"
    chat_max_tokens: int = 256
    chat_system: str = DEFAULT_CHAT_SYSTEM

    tts_url: str = ""
    tts_model: str = "omnivoice"
    tts_out_dir: str = "out"
    play: bool = False
    compare: bool = False
    omni_instructions: str = "female, child, high pitch, australian accent"
    omni_seed: int = 58842
    tts_voice: str = "bench"
    say_max_len: int = 140
    say_first_max: int = 60
    say_min_len: int = 12
    say_gap_ms: int = 90
    say_fade_ms: int = 8

    stt_url: str = ""
    stt_model: str = "Qwen/Qwen3-ASR-0.6B"
    warm_budget: int = 300

    @field_validator("llm_url", "tts_url", "stt_url", mode="before")
    @classmethod
    def _strip_url(cls, value: str | None) -> str:
        return strip_url(value)

    @field_validator("tts_model")
    @classmethod
    def _normalize_tts_model(cls, value: str) -> str:
        return value.lower()

    @field_validator("play", "compare", mode="before")
    @classmethod
    def _blank_flag_is_false(cls, value: object) -> object:
        return blank_flag_is_false(value)

    @property
    def llm_chat_url(self) -> str:
        if not self.llm_url:
            return ""
        return f"{self.llm_url}/v1/chat/completions"

    @property
    def tts_speech_url(self) -> str:
        if not self.tts_url:
            return ""
        return f"{self.tts_url}/v1/audio/speech"

    @property
    def tts_on(self) -> bool:
        return bool(self.tts_url)

    @property
    def stt_on(self) -> bool:
        return bool(self.stt_url)


settings = AppSettings()
