from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CHAT_SYSTEM = (
    "You are a friendly Australian helper for kids. Speak simply and warmly, "
    "in one or two short sentences."
)


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
    omni_instructions: str = "female, child, high pitch, australian accent"
    omni_seed: int = 58842
    clause_max_len: int = 140
    clause_first_max: int = 60
    clause_min_len: int = 12

    stt_url: str = ""
    stt_model: str = "Qwen/Qwen3-ASR-0.6B"
    warm_budget: int = 300

    @field_validator("llm_url", "tts_url", "stt_url", mode="before")
    @classmethod
    def _strip_url(cls, value: str | None) -> str:
        return (value or "").rstrip("/")


settings = AppSettings()
