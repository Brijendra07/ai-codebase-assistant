from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Codebase Assistant"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "fallback"
    llm_model_name: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None
    vertex_api_key: str | None = None
    vertex_base_url: str = "https://aiplatform.googleapis.com/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
