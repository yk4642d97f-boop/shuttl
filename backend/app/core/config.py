from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    public_base_url: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://shuttlevision:password@localhost:5432/shuttlevision"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    # Aliyun OSS
    oss_access_key_id: str = ""
    oss_access_key_secret: str = ""
    oss_bucket_name: str = "shuttlevision-videos"
    oss_endpoint: str = "https://oss-cn-hangzhou.aliyuncs.com"
    oss_region: str = "cn-hangzhou"

    # Video storage: "oss", "local", or "auto" (local unless real OSS keys are set)
    storage_backend: str = "auto"
    media_dir: str = str(BACKEND_DIR / "media")
    upload_token_expire_minutes: int = 120
    max_upload_bytes: int = 4 * 1024**3

    @property
    def oss_configured(self) -> bool:
        vals = (self.oss_access_key_id, self.oss_access_key_secret)
        return all(v and not v.strip().lower().startswith("your") for v in vals)

    @property
    def use_local_storage(self) -> bool:
        mode = self.storage_backend.strip().lower()
        if mode in ("local", "oss"):
            return mode == "local"
        return not self.oss_configured


settings = Settings()
