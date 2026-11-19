"""Config global — todo por env con prefijo VOYAGE_ (ver .env.example). 🌐"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOYAGE_", env_file=".env", extra="ignore")

    model_path: str = "./model"
    model_name: str = "voyage-4-nano"
    host: str = "127.0.0.1"
    port: int = 9009  # 9000 y algo ✅ (verificado libre)

    device: str = "cuda"  # "cuda" | "cpu" | "auto"
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"  # fallback auto: sdpa -> eager

    max_texts: int = 64
    max_chars: int = 32000
    max_tokens: int = 32768
    internal_batch_size: int = 32  # sub-batches en GPU (64 = 32+32)
    inference_timeout: float = 60.0
    lock_timeout: float = 0.0  # reservado (sin uso): el semáforo rechaza al momento, sin cola 💥
    rate_limit_per_minute: int = 120  # rate-limit suave local, solo /embed/* 🐢

    skip_model_load: bool = False  # True = modo validación sin GPU (tests) 🧪
