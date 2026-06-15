from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CHAT_SYSTEM = (
    "You are a friendly Australian helper for kids. Speak simply and warmly, "
    "in one or two short sentences."
)

# Map the full language name (OmniVoice TTS wants "English") to its ISO 639-1 code
# (Qwen3-ASR wants "en"). Codes per Qwen3-ASR table; names per OmniVoice docs/languages.md.
LANGUAGE_ISO = {
    "English": "en",
    "Vietnamese": "vi",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Portuguese": "pt",
    "Italian": "it",
    "Russian": "ru",
    "Thai": "th",
    "Indonesian": "id",
}


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    # Shared language: drives TTS `language`, STT `language` (as ISO code), and a
    # "reply in <language>" directive on the chat prompt.
    language: str = "English"

    llm_url: str = ""
    llm_model: str = "Qwen/Qwen3.5-4B"
    chat_max_tokens: int = 256
    chat_system: str = DEFAULT_CHAT_SYSTEM

    tts_url: str = ""
    # Accent/dialect attributes only apply to English/Chinese; leave empty (or clone)
    # for other languages. See docs/voice-design.md.
    omni_instructions: str = "female, child, high pitch, australian accent"
    omni_seed: int = 58842
    clause_max_len: int = 140

    stt_url: str = ""
    stt_model: str = "Qwen/Qwen3-ASR-0.6B"

    warm_budget: int = 300  # /api/warm health-poll budget; covers a first build
    cold_start_budget: int = 90  # per-request 503 tolerance; covers snapshot restore

    @field_validator("llm_url", "tts_url", "stt_url", mode="before")
    @classmethod
    def _strip_url(cls, value: str | None) -> str:
        return (value or "").rstrip("/")

    @field_validator("language", mode="before")
    @classmethod
    def _norm_language(cls, value: str | None) -> str:
        return (value or "English").strip() or "English"

    @property
    def stt_language(self) -> str | None:
        """ISO 639-1 code Qwen3-ASR expects, or None to let it auto-detect."""
        return LANGUAGE_ISO.get(self.language)

    @property
    def chat_system_prompt(self) -> str:
        """System prompt with a reply-language directive for non-English languages."""
        if self.language == "English":
            return self.chat_system
        return f"{self.chat_system} Always reply in {self.language}."

    @property
    def stt_transcriptions_url(self) -> str:
        return self.stt_url + "/v1/audio/transcriptions"

    @property
    def llm_chat_url(self) -> str:
        return self.llm_url + "/v1/chat/completions"

    @property
    def tts_speech_url(self) -> str:
        return self.tts_url + "/v1/audio/speech"


settings = AppSettings()
