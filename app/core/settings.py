from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
    )


def strip_url(value: str | None) -> str:
    return (value or "").rstrip("/")


def blank_flag_is_false(value: object) -> object:
    return False if value in (None, "") else value
