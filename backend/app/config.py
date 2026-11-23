from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite:///./restaurant_ai.db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    meta_app_secret: str = ""
    meta_verify_token: str = "restaurant-ai-dev"
    meta_access_token: str = ""
    meta_phone_number_id: str = ""
    meta_graph_version: str = "v21.0"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    rate_limit_per_minute: int = 120

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
